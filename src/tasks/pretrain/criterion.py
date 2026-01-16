import copy
import random
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

try:
    from xformers.ops import cross_entropy

    def lossfunc(t, s, temp):
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)

except ImportError:

    def lossfunc(t, s, temp):
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class MAELoss(nn.Module):
    """Loss function for Masked Autoencoder (MAE).

    Computes reconstruction loss in masked regions using specified criterion.

    Args:
        criterion: Loss criterion to use. One of ["l1", "l2", "smooth_l1"]. Defaults to "smooth_l1".
    """

    def __init__(
        self,
        criterion: str = "smooth_l1",
        weight: float = 1.0,
        eps: float = 1e-6,
        reduce: bool = True,
    ) -> None:
        super().__init__()
        # Initialize the loss criterion without reduction
        if criterion == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif criterion == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        else:  # smooth_l1
            self.criterion = nn.SmoothL1Loss(reduction="none")

        self.weight = weight
        self.eps = eps
        self.reduce = reduce

    def forward(
        self,
        pred_patch_tokens: torch.Tensor,  # [B, N, D]
        target_patch: torch.Tensor,  # [B, N, D]
        mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Args:
            pred_patch_tokens: Predicted patch embeddings of shape (batch, num_patches, dim)
            target_patch: Ground truth patch embeddings of shape (batch, num_patches, dim)
            mask: Binary mask indicating which patches are masked (1 = masked) of shape (batch, num_patches)

        Returns:
            If reduce=True: Scalar reconstruction loss averaged over masked patches.
            If reduce=False: [batch, num_patches] loss per patch.
        """
        # Compute per-patch loss: criterion returns [B, ..., D], then average over feature dimension -> [B, ...]
        loss = self.criterion(pred_patch_tokens, target_patch.expand_as(pred_patch_tokens)).mean(
            dim=-1
        )

        # Apply mask and weight
        loss *= mask * self.weight

        if self.reduce:
            # Sum losses and divide by the number of masked patches (clamped for stability)
            loss = loss.sum()
            num_masked = mask.sum().clamp_min(1)
            return loss / num_masked

        # Return per-patch losses when reduce=False
        return loss


class DINOLoss(nn.Module):
    """DINO loss function module.

    Computes loss for knowledge distillation between teacher and student networks.

    Args:
        student_temp (float): Student network temperature parameter. Defaults to 0.1
        weight (float): Loss weight. Defaults to 1.0
    """

    def __init__(
        self,
        student_temp: float = 0.1,
        weight: float = 1.0,
        reduce: bool = True,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.weight = weight
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.reduce = reduce

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output: torch.Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> torch.Tensor:
        """Applies Sinkhorn-Knopp algorithm to center teacher output.

        Args:
            teacher_output (torch.Tensor): Teacher network output logits
            teacher_temp (float): Teacher network temperature parameter
            n_iterations (int): Number of Sinkhorn-Knopp iterations. Defaults to 3

        Returns:
            torch.Tensor: Centered teacher output
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_output_list: List[torch.Tensor],
        teacher_out_softmaxed_centered_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes cross-entropy between teacher and student network outputs.

        Args:
            student_output_list (List[torch.Tensor]): List of student network outputs
            teacher_out_softmaxed_centered_list (List[torch.Tensor]): List of centered teacher network outputs

        Returns:
            torch.Tensor: Computed loss value
        """
        if self.weight == 0:
            return 0

        # if torch.isnan(student_output_list[0]).any():
        #     print("student_output_list[0] is nan")

        # if torch.isnan(teacher_out_softmaxed_centered_list[0]).any():
        #     print("teacher_out_softmaxed_centered_list[0] is nan")

        loss = torch.empty((student_output_list[0].shape[0], len(student_output_list)))
        for st_i, s in enumerate(student_output_list):
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            _loss = 0
            for t in teacher_out_softmaxed_centered_list:
                _loss += -torch.sum(t * lsm, dim=-1)

            loss[:, st_i] = _loss / len(teacher_out_softmaxed_centered_list)

        weigthed_loss = loss * self.weight

        if self.reduce:
            return weigthed_loss.mean()
        return weigthed_loss


class iBOTLoss(nn.Module):
    """iBOT loss function module.

    Args:
        student_temp (float): Student network temperature parameter. Defaults to 0.1
        weight (float): Loss weight. Defaults to 1.0
    """

    def __init__(
        self,
        student_temp: float = 0.1,
        weight: float = 1.0,
        reduce: bool = True,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.weight = weight
        self.reduce = reduce

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: torch.Tensor,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Applies Sinkhorn-Knopp algorithm to center teacher output.

        Args:
            teacher_output (torch.Tensor): Teacher network output logits
            teacher_temp (float): Teacher network temperature parameter
            n_masked_patches_tensor (torch.Tensor): Number of masked patches
            n_iterations (int): Number of Sinkhorn-Knopp iterations. Defaults to 3

        Returns:
            torch.Tensor: Centered teacher output
        """
        teacher_output = teacher_output.float()
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_patch_tokens: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
        student_masks_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy between patch tokens of teacher and student networks.

        Args:
            student_patch_tokens (torch.Tensor): Student network patch tokens (B, N, D)
            teacher_patch_tokens (torch.Tensor): Teacher network patch tokens (B, N, D)
            student_masks_flat (torch.Tensor): Student masks (B, N)
        Returns:
            torch.Tensor: Computed loss value
        """
        if self.weight == 0 or student_masks_flat.sum() == 0:
            return 0

        t = teacher_patch_tokens
        s = student_patch_tokens

        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = -loss * student_masks_flat.float()

        weigthed_loss = loss * self.weight

        if self.reduce:
            return weigthed_loss.mean()
        return weigthed_loss


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropy regularization loss.

    Loss function from Sablayrolles et al. - 2018 - Spreading vectors for similarity search paper.

    Args:
        weight (float): Loss weight. Defaults to 1.0
    """

    def __init__(
        self,
        weight: float = 1.0,
        reduce: bool = True,
    ) -> None:
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.weight = weight
        self.reduce = reduce

    def pairwise_NNs_inner(self, x: torch.Tensor) -> torch.Tensor:
        """Computes pairwise nearest neighbors for L2 normalized vectors.

        Uses PyTorch instead of Faiss to keep operations on GPU.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Nearest neighbor indices
        """
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Performs forward pass.

        Args:
            student_output (torch.Tensor): Student network output (BxD)
            eps (float): Epsilon for numerical stability. Defaults to 1e-8

        Returns:
            torch.Tensor: Computed loss value
        """
        if self.weight == 0:
            return 0

        with torch.amp.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps)
            assert not torch.isnan(
                loss
            ).any(), f"student_output: {student_output}, I: {I}, distances: {distances}"

        weigthed_loss = loss.clamp(min=0) * self.weight

        if self.reduce:
            return weigthed_loss.mean()
        return weigthed_loss


def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See `pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note `size_average`
            and `reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override `reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


class FocalLoss(_Loss):
    """Focal Loss implementation for classification tasks.

    Focal Loss is designed to address class imbalance by focusing training on hard examples.
    It adds a modulating factor to the standard cross entropy loss to reduce the relative loss
    for well-classified examples and increase the loss for hard examples.

    Args:
        mode (str): Loss mode - 'binary', 'multiclass' or 'multilabel'
        alpha (float, optional): Prior probability of having positive value in target.
            Helps balance positive/negative examples. Defaults to None.
        gamma (float, optional): Power factor for dampening weight (focal strength).
            Higher values focus more on hard examples. Defaults to 2.0.
        ignore_index (int, optional): If not None, targets may contain values to be ignored.
            Target values equal to ignore_index will be ignored from loss computation. Defaults to None.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. Defaults to "mean".
        normalized (bool, optional): Whether to compute normalized focal loss as described in
            https://arxiv.org/pdf/1909.07829.pdf. Defaults to False.
        reduced_threshold (float, optional): Switch to reduced focal loss. When using this mode,
            reduction should be set to "sum". Defaults to None.

    Shape:
        - y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W)
        - y_true (torch.Tensor): Ground truth tensor of shape (N, H, W) or (N, C, H, W)

    Reference:
        https://github.com/BloodAxe/pytorch-toolbelt
    """

    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Focal Loss.

        Args:
            y_pred (torch.Tensor): Predicted tensor
            y_true (torch.Tensor): Ground truth tensor

        Returns:
            torch.Tensor: Computed focal loss value
        """
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    dice_score = soft_tversky_score(output, target, 0.5, 0.5, smooth, eps, dims)
    return dice_score


def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """Tversky loss

    References:
        https://arxiv.org/pdf/2302.05666
        https://arxiv.org/pdf/2303.16296

    """
    assert output.size() == target.size()

    if dims is not None:
        output_sum = torch.sum(output, dim=dims)
        target_sum = torch.sum(target, dim=dims)
        difference = LA.vector_norm(output - target, ord=1, dim=dims)
    else:
        output_sum = torch.sum(output)
        target_sum = torch.sum(target)
        difference = LA.vector_norm(output - target, ord=1)

    intersection = (output_sum + target_sum - difference) / 2  # TP
    fp = output_sum - intersection
    fn = target_sum - intersection

    tversky_score = (intersection + smooth) / (
        intersection + alpha * fp + beta * fn + smooth
    ).clamp_min(eps)
    return tversky_score


class DiceLoss(_Loss):
    """Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases.

    This loss function is commonly used in image segmentation tasks to measure the overlap between
    predicted and ground truth masks. It can handle binary, multiclass, and multilabel segmentation
    scenarios.

    The Dice coefficient is calculated as:
        dice = (2 * intersection) / (sum of predictions + sum of ground truth)

    Attributes:
        mode (str): Loss mode ('binary', 'multiclass' or 'multilabel')
        classes (Optional[List[int]]): List of classes that contribute in loss computation
        from_logits (bool): Whether input is raw logits
        smooth (float): Smoothness constant for dice coefficient
        eps (float): Small epsilon for numerical stability
        log_loss (bool): Whether to compute loss as -log(dice_coeff)
        ignore_index (Optional[int]): Label that indicates ignored pixels
        reduction (bool): Whether to reduce the loss
    """

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        reduction: bool = True,
    ):
        """Initialize Dice loss.

        Args:
            mode (str): Loss mode 'binary', 'multiclass' or 'multilabel'
            classes (Optional[List[int]], optional): List of classes that contribute in loss computation.
                By default, all channels are included.
            log_loss (bool, optional): If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`.
                Defaults to False.
            from_logits (bool, optional): If True, assumes input is raw logits. Defaults to True.
            smooth (float, optional): Smoothness constant for dice coefficient (a). Defaults to 0.0.
            ignore_index (Optional[int], optional): Label that indicates ignored pixels (does not contribute to loss).
                Defaults to None.
            eps (float, optional): A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps). Defaults to 1e-7.
            reduction (bool, optional): Whether to reduce the loss. Defaults to True.

        Shape:
            - **y_pred** (torch.Tensor): Tensor of shape (N, C, H, W)
            - **y_true** (torch.Tensor): Tensor of shape (N, H, W) or (N, C, H, W)

        Reference:
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Dice loss.

        Args:
            y_pred (torch.Tensor): Predicted tensor
            y_true (torch.Tensor): Ground truth tensor

        Returns:
            torch.Tensor: Computed Dice loss
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        if self.reduction:
            return self.aggregate_loss(loss)
        return loss

    def aggregate_loss(self, loss):
        """Aggregate the loss values.

        Args:
            loss (torch.Tensor): Loss tensor to aggregate

        Returns:
            torch.Tensor: Mean of the loss values
        """
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        """Compute the Dice score.

        Args:
            output (torch.Tensor): Predicted tensor
            target (torch.Tensor): Ground truth tensor
            smooth (float, optional): Smoothness constant. Defaults to 0.0.
            eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-7.
            dims (tuple, optional): Dimensions to reduce over. Defaults to None.

        Returns:
            torch.Tensor: Computed Dice score
        """
        return soft_dice_score(output, target, smooth, eps, dims)


class PCGrad:
    def __init__(self, optimizer, reduction="mean"):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """

        return self._optim.step()

    def pc_backward(self, objectives, lightning_module):
        """
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        """

        grads, shapes, has_grads = self._pack_grad(objectives, lightning_module)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """

        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, lightning_module):
        """
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grads, shapes, has_grads = [], [], []
        for idx, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            # obj.backward(retain_graph=True)
            lightning_module.manual_backward(obj, retain_graph=(idx < len(objectives) - 1))
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


if __name__ == "__main__":
    logits = torch.randn(10, 10)
    labels = torch.randint(0, 10, (10,))
    loss = nn.CrossEntropyLoss()(logits, labels)
    print(loss)
