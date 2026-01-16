import copy
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import einops
import kornia.augmentation as K
import numpy as np
import rootutils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning import pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from timm.models.layers import Mlp
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tasks.pretrain.criterion import DiceLoss, FocalLoss
from src.tasks.pretrain.utils import (
    LoggingUtil,
    combine_images_and_heatmaps,
    compute_pca_features,
    convert_attn_to_heatmap,
    cosine_scheduler,
    min_max_scale_1d,
    min_max_scale_2d,
    patchify,
    resize_attention_map,
    unpatchify,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BaseModule(pl.LightningModule):
    """Base module for self-supervised learning.

    This module implements the basic functionality of self-supervised learning models.
    It handles the training loop, validation, test, and optimization.

    Args:
        encoder (torch.nn.Module): Encoder network to extract image features
        decoder (torch.nn.Module): Decoder network to reconstruct features from encoder
        target_generator (torch.nn.Module): Network to generate training targets
        criterion (torch.nn.Module): Loss function used for model training
        optimizer (torch.optim.Optimizer): Optimizer to optimize model parameters
        gradient_accumulation_steps (int): Number of batches to accumulate gradients
        gradient_clip_val (float): Max threshold for gradient clipping
        compile (bool): Whether to use PyTorch 2.0 compile feature
        batch_size (int): Batch size
        num_devices (int): Number of GPU devices to use for training
        num_nodes (int): Number of nodes to use for distributed training
        wd_schedule (Callable, optional): Weight decay scheduler
        lr_schedule (Callable, optional): Learning rate scheduler
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        target_generator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_devices: int,
        num_nodes: int,
        compile: bool,
        criterion: Optional[torch.nn.Module] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        wd_schedule: Optional[Callable] = None,
        lr_schedule: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "encoder",
                "target_generator",
                "decoder",
                "criterion",
            ],
        )
        self.encoder = encoder
        self.decoder = decoder
        self.target_generator = target_generator
        self.criterion = criterion
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        # validation data
        self.val_data_to_visualize = None
        self.train_data_to_visualize = None
        self.target_class_list = [
            "AdvancedDiabeticRetinopathy",
            "CentralRetinalVeinOcclusion",
            "BranchRetinalVeinOcclusion",
            "DryAge-relatedMacularDegeneration",
            "GlaucomatousDiscChange",
        ]
        self.class_counters = {cls: 0 for cls in self.target_class_list}
        self.wd_schedule = None
        self.lr_schedule = None
        self.steps_per_epoch = None
        self.num_train_data = None
        self.logging_util = LoggingUtil()

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Dict[str, Any]): Input dictionary containing image tensor under 'img' key

        Returns:
            torch.Tensor: Encoded latent representation
        """
        imgs = x["img"]
        res = self.forward_ssl_encoder(imgs, masking=False)
        out = self.forward_decoder(res["features"])
        return out

    def forward_ssl_encoder(
        self, x: torch.Tensor, masking: bool = True, fixed_mask: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the SSL encoder.

        Args:
            x (torch.Tensor): Input image tensor
            masking (bool, optional): Whether to apply masking. Defaults to True.
            fixed_mask (bool, optional): Whether to use fixed mask. Defaults to False.
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - features: Latent representation
                - mask: Mask
                - restore_ids: Indices to restore
        """
        res = self.encoder(x, masking=masking, fixed_mask=fixed_mask)
        return res

    def forward_decoder(
        self, features: torch.Tensor, ids_restore_mim: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            features (torch.Tensor): Input feature tensor
            ids_restore_mim (Optional[torch.Tensor], optional): Indices for restoring original order.
                Defaults to None.

        Returns:
            torch.Tensor: Decoded output
        """
        if ids_restore_mim is not None:
            return self.decoder(features, ids_restore_mim)

        return self.decoder(features)

    def shared_step(self, batch: Dict[str, Any], batch_idx: int, split="train") -> torch.Tensor:
        """Shared step for training, validation and testing.

        To be implemented by child classes.

        Args:
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index
            split (str, optional): Dataset split. Defaults to "train".

        Returns:
            torch.Tensor: Loss value
        """
        return NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss value
        """
        loss = self.shared_step(batch, batch_idx, split="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss value
        """
        loss = self.shared_step(batch, batch_idx, split="val")
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch (Dict[str, Any]): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss value
        """
        loss = self.shared_step(batch, batch_idx, split="test")
        return loss

    def on_train_start(self):
        torch.autograd.set_detect_anomaly(True)

    def on_train_epoch_start(self):
        """Method called when the training epoch begins.

        Initializes defaultdict.
        """
        self.train_data_to_visualize = None

    def on_validation_epoch_start(self):
        """Method called when the validation epoch begins.

        Logs current epoch number and initializes defaultdict
        for visualizing validation data.
        """
        self.val_data_to_visualize = defaultdict(list)

    def setup(self, stage: str) -> None:
        """Setup for each stage.

        Compiles model if compile=True and stage='fit'

        Args:
            stage (str): Current stage ('fit' or 'test')
        """
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")

        if isinstance(self.encoder, dict) or hasattr(self.encoder, "__getitem__"):
            encoder_dict = {}
            for k, v in self.encoder.items():
                encoder_dict[k] = (
                    torch.compile(v) if self.hparams.compile and stage == "fit" else v
                )
            self.encoder = nn.ModuleDict(encoder_dict)

        else:
            self.encoder = (
                torch.compile(self.encoder)
                if self.hparams.compile and stage == "fit"
                else self.encoder
            )

        if isinstance(self.decoder, dict) or hasattr(self.decoder, "__getitem__"):
            decoder_dict = {}
            for k, v in self.decoder.items():
                decoder_dict[k] = (
                    torch.compile(v) if self.hparams.compile and stage == "fit" else v
                )
            self.decoder = nn.ModuleDict(decoder_dict)

        else:
            self.decoder = (
                torch.compile(self.decoder)
                if self.hparams.compile and stage == "fit"
                else self.decoder
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and optional scheduler config
        """
        # Separate parameters that do not apply weight decay
        decay = []
        no_decay = []
        for name, param in self.trainer.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(
                nd in name
                for nd in [
                    "pos_embed",
                    "cls_token",
                    "reg_token",
                    "mask_token",
                    "norm",
                    "bias",
                ]
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        # Parameter group settings
        parameters = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        optimizer = self.hparams.optimizer(params=parameters)

        self.num_train_data = len(self.trainer.datamodule.data_train)
        self.steps_per_epoch = math.ceil(
            self.num_train_data
            / (self.hparams.batch_size * self.hparams.num_devices * self.hparams.num_nodes)
        )

        if self.hparams.wd_schedule is not None:
            self.wd_schedule = self.hparams.wd_schedule(niter_per_ep=self.steps_per_epoch)

        if self.hparams.lr_schedule is not None:
            lr = optimizer.param_groups[0]["lr"]
            self.lr_schedule = self.hparams.lr_schedule(
                base_value=lr,
                niter_per_ep=self.steps_per_epoch,
            )
        return {"optimizer": optimizer}

    @rank_zero_only
    def _visualize_attn(self):
        """Visualize attention maps.

        Args:
            class_token_only (bool, optional): Whether to only use class token. Defaults to False.
        """

        concat_class_batch, concat_patch_batch = self._prepare_attn_images()
        for i, concat_class in enumerate(concat_class_batch):
            self.logging_util.log_image_tensor(self, concat_class, f"attn_class/{i}")

        for i, concat_patch in enumerate(concat_patch_batch):
            self.logging_util.log_image_tensor(self, concat_patch, f"attn_patch/{i}")

    @torch.inference_mode()
    @rank_zero_only
    def _visualize_avg_register_features(self):
        """Visualizes the average of register features.

        Calculates and visualizes the average of register features for 4 randomly selected images
        from the validation dataset.
        """
        images = self.val_data_to_visualize["img"]
        images = torch.cat(images, dim=0)
        idx = torch.randperm(len(images))[:3]
        images = images[idx]
        features = self.encoder.forward_features(images)["features"][:, : self.num_prefix_tokens]

        avg_register_features = features.mean(dim=0)  # (R, D)
        avg_register_features = avg_register_features.repeat_interleave(5, dim=0)
        avg_register_features = min_max_scale_1d(avg_register_features)

        heat_map = convert_attn_to_heatmap(avg_register_features, "magma")
        self.logging_util.log_image_tensor(self, heat_map, "val/avg_register_features")

    @torch.inference_mode()
    @rank_zero_only
    def _visualize_patch_feature_norm(self):
        """Visualizes L2 norm of patch features.

        Calculates and visualizes L2 norm of patch features for 4 randomly selected images
        from the validation dataset.
        Displays original images and heatmaps side by side.
        """
        images = self.val_data_to_visualize["img"]
        images = torch.cat(images, dim=0)
        idx = torch.randperm(len(images))[:3]
        images = images[idx]

        # Extract intermediate features and calculate L2 norm
        features = self.encoder.get_intermediate_features(images)
        features_2d = []
        for k, feature in features.items():
            if k == "patch":
                continue

            # Calculate L2 norm of patch features (B, N)
            feature_norm = torch.norm(feature[:, self.num_prefix_tokens :], dim=2)

            # Convert to 2D shape and upsample
            H = int(np.sqrt(feature_norm.size(1)))
            feature_norm = (
                feature_norm.reshape(-1, H, H)
                .repeat_interleave(self.encoder.patch_size, dim=1)
                .repeat_interleave(self.encoder.patch_size, dim=2)
            )

            # Normalize
            features_2d.append(min_max_scale_2d(feature_norm.unsqueeze(1)).squeeze(1))

        # Concatenate feature maps and create margin
        features_2d = torch.cat(features_2d, dim=2)  # (B, H, 3*W)
        margin = torch.ones((3, features_2d.shape[1], 5))

        concat_list = []
        for i in range(features_2d.size(0)):
            heat_map = convert_attn_to_heatmap(features_2d[i])
            concat = torch.cat([images[i].to("cpu"), margin, heat_map], dim=2)
            concat_list.append(concat)

        self.logging_util.log_image_tensor(self, torch.cat(concat_list, dim=1), "feature_norm")

    @torch.inference_mode()
    @rank_zero_only
    def _visualize_patch_pca_features(self):
        """Visualizes PCA analysis results of patch features.

        Performs and visualizes PCA analysis of patch features for 4 randomly selected images
        from the validation dataset.
        Displays original images and PCA feature maps side by side.
        """
        # Prepare image data
        images = self.val_data_to_visualize["img"]
        images = torch.cat(images, dim=0)[torch.randperm(len(images))[:3]]

        # Feature extraction
        # Extract features and separate patch features
        feats = self.encoder.get_intermediate_features(images)["late"]  # (B, N+R, D)
        patch_feature = feats[:, self.num_prefix_tokens :]  # (B, N, D)
        B, N, D = patch_feature.shape

        # Calculate PCA features - Process by batch
        pca_feats = []
        for features in patch_feature:
            pf = compute_pca_features(features, None)  # (N, 3)
            pca_feats.append(pf)
        pca_feats = torch.stack(pca_feats)  # (B, N, 3)

        # Resize feature map
        h = w = int(np.sqrt(N))
        pca_feats = einops.rearrange(pca_feats, "b (h w) c -> b h w c", h=h, w=w).permute(
            0, 3, 1, 2
        )  # (B, 3, h, w)

        # Upsampling
        target_size = (h * self.encoder.patch_size, w * self.encoder.patch_size)
        pca_feats = torch.nn.functional.interpolate(
            pca_feats, size=target_size, mode="nearest"
        )  # (B, 3, H, W)

        # Margin and image concatenation for visualization
        margin = torch.ones((3, pca_feats.shape[2], 5))
        concat = torch.cat(
            [
                torch.cat([img, margin, feat], dim=2)
                for img, feat in zip(images.to("cpu"), pca_feats)
            ],
            dim=1,
        )

        self.logging_util.log_image_tensor(self, concat.to("cpu"), "pca_features")

    def _prepare_attn_images(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare attention visualization images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated visualization tensor
        """
        # Select samples and compute attention
        images = self.val_data_to_visualize["img"]
        images = torch.cat(images, dim=0)
        idx = torch.randperm(len(images))[:3]
        images = images[idx]

        concat_class_list = []
        concat_patch_list = []

        # Calculate attention maps at once
        attention_maps = self.encoder.get_self_attention(images)

        for _, attn in attention_maps.items():
            # Extract class and patch attention
            class_attn = attn[:, :, 0, self.num_prefix_tokens :]
            # Min-max normalization for class attention
            min_vals_cls = class_attn.min(dim=-1, keepdim=True).values
            max_vals_cls = class_attn.max(dim=-1, keepdim=True).values
            class_attn = (class_attn - min_vals_cls) / (max_vals_cls - min_vals_cls + 1e-6)

            patch_attn = attn[:, :, self.num_prefix_tokens :, self.num_prefix_tokens :]
            # Min-max normalization for patch attention
            min_vals_patch = patch_attn.min(dim=-1, keepdim=True).values
            max_vals_patch = patch_attn.max(dim=-1, keepdim=True).values
            patch_attn = (patch_attn - min_vals_patch) / (max_vals_patch - min_vals_patch + 1e-6)
            patch_attn = patch_attn.mean(dim=2)

            # Resize attention map
            class_attn_interleaved = resize_attention_map(
                class_attn, images.shape[2:]
            )  # (B, HD, H, W)
            patch_attn_interleaved = resize_attention_map(
                patch_attn, images.shape[2:]
            )  # (B, HD, H, W)

            # Calculate margin
            margin = torch.ones((3, patch_attn_interleaved.shape[2], 5))

            # Combine image and heatmap
            concat_class = combine_images_and_heatmaps(
                images, class_attn_interleaved, margin
            )  # (B, 3, H, W+W*HD)
            concat_patch = combine_images_and_heatmaps(
                images, patch_attn_interleaved, margin
            )  # (B, 3, H, W+W*HD)

            concat_class_list.append(concat_class)
            concat_patch_list.append(concat_patch)

        return torch.cat(concat_class_list, dim=2), torch.cat(concat_patch_list, dim=2)

    @torch.inference_mode()
    def _log_grad_norm(
        self,
        parameters: Union[Iterable[torch.nn.Parameter], torch.nn.Parameter],
        norm_type: int = 2,
        split: str = "train",
        prefix: str = "",
    ) -> float:
        """Calculates and logs gradient norm."""
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm = torch.zeros(1, device=self.device)

        try:
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.pow(norm_type)
            total_norm = total_norm.pow(1.0 / norm_type)
        except Exception as e:
            print(f"⚠️ Error occurred during gradient norm calculation: {e}")
            total_norm = torch.tensor(float("nan"), device=self.device)

        if prefix:
            title = f"{split}/{prefix}_grad_norm"
        else:
            title = f"{split}/grad_norm"

        self.log(
            title,
            total_norm.item(),
            sync_dist=True,
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    @torch.inference_mode()
    def _log_parameter_norm(
        self,
        parameters: Union[Iterable[torch.nn.Parameter], torch.nn.Parameter],
        norm_type: int = 2,
        split: str = "train",
    ) -> float:
        """Calculates and logs parameter norm."""
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]

        total_norm = torch.zeros(1, device=self.device)

        try:
            for p in parameters:
                param_norm = p.data.norm(norm_type)
                total_norm += param_norm.pow(norm_type)
            total_norm = total_norm.pow(1.0 / norm_type)
        except Exception as e:
            print(f"⚠️ Error occurred during parameter norm calculation: {e}")
            total_norm = torch.tensor(float("nan"), device=self.device)

        title = f"{split}/param_norm"
        self.log(
            title,
            total_norm.item(),
            sync_dist=True,
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    def _get_lesion_data(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Get lesion data for visualization."""
        img = batch["img"]
        label = batch["label"]
        class_list = batch["class_name_list"]
        dataset_name = batch["dataset_name"]
        lesion_data = defaultdict(list)

        for dataset, l, im in zip(dataset_name, label, img):
            _idx = {
                class_list[dataset].index(target_class): target_class
                for target_class in self.target_class_list
                if target_class in class_list[dataset]
            }
            _target_label = l[list(_idx.keys())]

            # Collect only up to 10 per class
            if _target_label.eq(1).any():
                # Check classes the current sample belongs to
                for idx, target_class in _idx.items():
                    if l[idx] == 1:
                        if self.class_counters[target_class] >= 10:
                            continue

                        self.class_counters[target_class] += 1
                        lesion_data["img"].append(im.unsqueeze(0))
                        lesion_data["label"].append(target_class)
                        lesion_data["dataset_name"].append(dataset)
                        break

        # Convert to tensor
        for k, v in lesion_data.items():
            if k not in ["img"]:
                continue
            lesion_data[k] = torch.cat(v, dim=0)

        return dict(lesion_data)


class RetMAEModule(BaseModule):
    """RetMAE (Retinal Masked Autoencoder) Module.

    Implements training and inference for Retinal Masked Autoencoder model, with support for
    global context distillation and high-frequency MI.

    Args:
        encoder (torch.nn.Module): Encoder network
        decoder (torch.nn.Module): Decoder network
        target_generator (torch.nn.Module): Target generator
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        gradient_accumulation_steps (int): Number of gradient accumulation steps
        gradient_clip_val (float): Gradient clipping value
        compile (bool): Whether to compile model
        batch_size (int): Batch size
        num_devices (int): Number of devices
        num_nodes (int): Number of nodes
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        target_generator: torch.nn.Module,
        criterion: Optional[torch.nn.Module] = None,
        mi_max_high_freq: Optional[Callable] = None,
        global_context: Optional[Callable] = None,
        optimizer_main: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 128,
        num_devices: int = 1,
        num_nodes: int = 1,
        compile: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        wd_schedule_main: Callable = None,
        lr_schedule_main: Callable = None,
        optimizer_distill: Optional[torch.optim.Optimizer] = None,
        optimizer_hmi: Optional[torch.optim.Optimizer] = None,
        wd_schedule_distill: Optional[Callable] = None,
        wd_schedule_hmi: Optional[Callable] = None,
        lr_schedule_distill: Optional[Callable] = None,
        lr_schedule_hmi: Optional[Callable] = None,
        momentum_schedule: Optional[Callable] = None,
        lambda_rec: float = 1,
        lambda_distill: float = 1,
        lambda_hmi: float = 1,
        high_freq_start_epoch: int = 20,
        module_ckpt_path: Optional[str] = None,
        teacher_temp: float = 0.07,
        student_temp: float = 0.1,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            target_generator=target_generator,
            optimizer=optimizer_main,
            batch_size=batch_size,
            num_devices=num_devices,
            num_nodes=num_nodes,
            compile=compile,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_val=gradient_clip_val,
        )
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "encoder",
                "target_generator",
                "decoder",
                "criterion",
                "criterion",
                "global_context",
                "mi_max_high_freq",
                "optimizer_main",
                "optimizer_distill",
                "optimizer_hmi",
                "wd_schedule_main",
                "lr_schedule_main",
                "wd_schedule_distill",
                "wd_schedule_hmi",
                "lr_schedule_distill",
                "lr_schedule_hmi",
                "momentum_schedule",
            ],
        )

        self.optimizer_main = optimizer_main
        self.optimizer_distill = optimizer_distill
        self.optimizer_hmi = optimizer_hmi
        self.wd_schedule_main = wd_schedule_main
        self.lr_schedule_main = lr_schedule_main
        self.wd_schedule_distill = wd_schedule_distill
        self.wd_schedule_hmi = wd_schedule_hmi
        self.lr_schedule_distill = lr_schedule_distill
        self.lr_schedule_hmi = lr_schedule_hmi

        self.mi_max_high_freq = mi_max_high_freq
        self.criterion = criterion

        # Get num_prefix_tokens safely from encoder backbone
        if hasattr(encoder.backbone, "num_prefix_tokens"):
            self.num_prefix_tokens = encoder.backbone.num_prefix_tokens
        elif hasattr(encoder.backbone, "model") and hasattr(
            encoder.backbone.model, "num_prefix_tokens"
        ):
            self.num_prefix_tokens = encoder.backbone.model.num_prefix_tokens
        else:
            self.num_prefix_tokens = getattr(encoder.backbone, "num_prefix_tokens", 0)

        self.wd_schedules = {}
        self.lr_schedules = {}
        self.lambda_rec = lambda_rec
        self.lambda_distill = lambda_distill
        self.lambda_hmi = lambda_hmi
        self.high_freq_start_epoch = high_freq_start_epoch
        self.momentum_schedule = momentum_schedule
        self.step_count = 0
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.augmentations = None

        self.global_context = global_context

        if self.lambda_distill > 0:
            if global_context is None:
                raise ValueError("global_context must be provided when lambda_distill > 0")

            gc_out_dim = getattr(global_context, "out_dim", self.encoder.backbone.out_dim)
            self.proj_distill_student = nn.Linear(self.encoder.backbone.out_dim, gc_out_dim)
            if self.hparams.compile:
                torch.compile(self.proj_distill_student)

            self.proj_distill_student.eval()

            # Freeze global context
            for p in self.global_context.parameters():
                p.requires_grad = False
            self.global_context.eval()

            if not self.lambda_hmi > 0:
                self.momentum_schedule = None
        else:
            self.proj_distill_student = None
            self.optimizer_distill = None
            self.wd_schedule_distill = None
            self.lr_schedule_distill = None

        if self.lambda_hmi > 0:
            self.is_hmi_train = False
            self.proj_hmi_student = nn.Sequential(
                nn.Linear(self.encoder.backbone.out_dim, self.encoder.backbone.out_dim),
                nn.LayerNorm(self.encoder.backbone.out_dim),
            )
            self.norm_hmi = nn.LayerNorm(self.encoder.backbone.out_dim)
            if self.hparams.compile:
                if self.mi_max_high_freq:
                    torch.compile(self.mi_max_high_freq)
                torch.compile(self.proj_hmi_student)
                torch.compile(self.norm_hmi)

            for p in self.proj_hmi_student.parameters():
                p.requires_grad = False

            for p in self.norm_hmi.parameters():
                p.requires_grad = False

            if self.mi_max_high_freq:
                for p in self.mi_max_high_freq.parameters():
                    p.requires_grad = False

            assert self.momentum_schedule is not None, "❌ momentum_schedule is required"
            self.ema_init = False

        else:
            self.mi_max_high_freq = None
            self.optimizer_hmi = None
            self.wd_schedule_hmi = None
            self.lr_schedule_hmi = None

    def _get_augmentations(self):
        aug = {}
        # train augmentations
        strong_aug = nn.Sequential(
            K.RandomAffine(
                degrees=(-5, 5),
                scale=(1.0, 1.0),
                translate=None,
                shear=None,
                padding_mode="zeros",
                p=1.0,
                same_on_batch=False,
                align_corners=False,
            ),
            K.RandomResizedCrop(
                size=(self.encoder.img_size, self.encoder.img_size),
                scale=(0.7, 1.0),
                ratio=(1, 1),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0),
        )
        aug["train"] = strong_aug

        # validation augmentations
        aug["val"] = nn.Sequential(
            K.Resize(self.encoder.img_size),
        )
        return aug

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.

        This method initializes the data augmentations for the current epoch by calling
        the `_get_augmentations` method and assigning the result to `self.augmentations`.
        It also calls the parent class's `on_train_epoch_start` to ensure any base
        functionality is executed.
        """
        super().on_train_epoch_start()
        self.augmentations = self._get_augmentations()

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Dict[str, Any]): Input dictionary containing image tensor under 'img' key

        Returns:
            torch.Tensor: Encoded latent representation
        """
        imgs = x["img"]
        res = self.forward_ssl_encoder(imgs, masking=False)
        out = self.forward_decoder(res["features"])
        return out

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""

        def _get_params(module, decay=None, no_decay=None, include_frozen=True):
            if decay is None:
                decay = []
            if no_decay is None:
                no_decay = []
            for name, param in module.named_parameters():
                if (not include_frozen) and (not param.requires_grad):
                    continue
                if any(
                    nd in name
                    for nd in ["pos_embed", "cls_token", "reg_token", "mask_token", "norm", "bias"]
                ):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        self.num_train_data = len(self.trainer.datamodule.data_train)
        self.steps_per_epoch = math.ceil(
            self.num_train_data
            / (self.hparams.batch_size * self.hparams.num_devices * self.hparams.num_nodes)
        )

        optimizer_configs = []

        # 1. Main Optimizer
        decay_main, no_decay_main = [], []
        _get_params(self.encoder, decay_main, no_decay_main)
        _get_params(self.decoder, decay_main, no_decay_main)

        param_main = [{"params": decay_main}, {"params": no_decay_main, "weight_decay": 0.0}]
        self._optimizer_main = self.optimizer_main(params=param_main)
        optimizer_configs.append({"optimizer": self._optimizer_main, "name": "main"})

        if self.wd_schedule_main:
            self.wd_schedules["main"] = self.wd_schedule_main(
                niter_per_ep=self.steps_per_epoch, epochs=self.trainer.max_epochs
            )
        if self.lr_schedule_main:
            self.lr_schedules["main"] = self.lr_schedule_main(
                base_value=self._optimizer_main.param_groups[0]["lr"],
                niter_per_ep=self.steps_per_epoch,
                epochs=self.trainer.max_epochs,
            )

        # 2. Distill Optimizer
        if self.optimizer_distill and self.proj_distill_student:
            decay_distill, no_decay_distill = _get_params(self.proj_distill_student)
            param_distill = [
                {"params": decay_distill},
                {"params": no_decay_distill, "weight_decay": 0.0},
            ]
            self._optimizer_distill = self.optimizer_distill(params=param_distill)
            optimizer_configs.append({"optimizer": self._optimizer_distill, "name": "distill"})

            if self.wd_schedule_distill:
                self.wd_schedules["distill"] = self.wd_schedule_distill(
                    niter_per_ep=self.steps_per_epoch, epochs=self.trainer.max_epochs
                )
            if self.lr_schedule_distill:
                self.lr_schedules["distill"] = self.lr_schedule_distill(
                    base_value=self._optimizer_distill.param_groups[0]["lr"],
                    niter_per_ep=self.steps_per_epoch,
                    epochs=self.trainer.max_epochs,
                )
        else:
            self._optimizer_distill = None

        # 3. HMI Optimizer
        if self.optimizer_hmi and self.lambda_hmi > 0:
            decay_hmi, no_decay_hmi = [], []
            if self.mi_max_high_freq:
                d, nd = _get_params(self.mi_max_high_freq)
                decay_hmi.extend(d)
                no_decay_hmi.extend(nd)
            if self.proj_hmi_student:
                d, nd = _get_params(self.proj_hmi_student)
                decay_hmi.extend(d)
                no_decay_hmi.extend(nd)
            if self.norm_hmi:
                d, nd = _get_params(self.norm_hmi)
                decay_hmi.extend(d)
                no_decay_hmi.extend(nd)

            param_hmi = [{"params": decay_hmi}, {"params": no_decay_hmi, "weight_decay": 0.0}]
            self._optimizer_hmi = self.optimizer_hmi(params=param_hmi)
            optimizer_configs.append({"optimizer": self._optimizer_hmi, "name": "hmi"})

            if self.wd_schedule_hmi:
                self.wd_schedules["hmi"] = self.wd_schedule_hmi(
                    niter_per_ep=self.steps_per_epoch, epochs=self.trainer.max_epochs
                )
            if self.lr_schedule_hmi:
                self.lr_schedules["hmi"] = self.lr_schedule_hmi(
                    base_value=self._optimizer_hmi.param_groups[0]["lr"],
                    niter_per_ep=self.steps_per_epoch,
                    epochs=self.trainer.max_epochs,
                    start_epoch=self.high_freq_start_epoch,
                )
        else:
            self._optimizer_hmi = None

        # Momentum Schedule
        if self.momentum_schedule:
            self.momentum_schedule = self.momentum_schedule(
                niter_per_ep=self.steps_per_epoch, epochs=self.trainer.max_epochs
            )

        return optimizer_configs

    def compute_loss_rec(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MAE (Masked Autoencoder) loss."""
        preds = input_tensor
        mask = (masks == 1).to(preds.dtype)
        loss_per_view = self.criterion(preds, target_tensor, mask).sum(dim=-1)
        loss_per_view /= mask.sum(dim=-1).clamp(min=1.0)
        return loss_per_view.mean()

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess input image."""
        return img.float() / 255.0 if img.max() > 1.0 else img

    def compute_loss(
        self,
        batch: Dict[str, Any],
        split: str = "train",
        do_distill: bool = False,
        do_hmi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for SSL training.

        Args:
            batch (Dict[str, Any]): Input batch containing images and metadata
            split (str, optional): Dataset split. Defaults to "train".
            do_distill (bool, optional): Whether to compute global MI loss. Defaults to False.
            do_hmi (bool, optional): Whether to compute high-freq MI loss. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing computed losses
        """
        # forward
        img = self.preprocess(batch["img"])
        x = self.augmentations[split](img)
        encoder_out = self.forward_ssl_encoder(x, masking=True, fixed_mask=(split == "val"))
        feats, masks, restore_ids = (
            encoder_out["features"],
            encoder_out["mask_mim"],
            encoder_out["restore_ids_mim"],
        )

        loss_dict = {}

        # MAE
        decoder_out = self.decoder(feats, restore_ids)
        patch_preds = decoder_out["patch"]
        loss_dict["mae"] = (
            self.lambda_rec
            * self.compute_loss_rec(patch_preds, self.target_generator(x), masks)
            / self.hparams.gradient_accumulation_steps
        )

        # high‑freq MI
        if do_hmi and self.mi_max_high_freq:
            if not self.ema_init:
                self.ema = ModelEMA(self.encoder.backbone)
                self.ema_init = True

            if not self.is_hmi_train:
                self.is_hmi_train = True
                self._enable_grads(self.mi_max_high_freq)
                self._enable_grads(self.proj_hmi_student)
                self._enable_grads(self.norm_hmi)

            with torch.no_grad():
                ema_features = self.ema.ema_model.forward_features(x, infer_high_freq=True)[
                    "features"
                ].detach()

            high_freq_latents = self.norm_hmi(ema_features[:, 0])
            student_latents = self.proj_hmi_student(feats[:, 0])
            loss_mi_hmi = self.mi_max_high_freq.learning_loss(student_latents, high_freq_latents)
            loss_dict["high_freq"] = (
                self.lambda_hmi * loss_mi_hmi / self.hparams.gradient_accumulation_steps
            )

        # global context distillation
        if do_distill:
            with torch.no_grad():
                global_context_latents = self.global_context.forward_features(img)[
                    "features"
                ].detach()

                # Handling RetCLIP's 3D output (B, 1, D)
                if global_context_latents.dim() == 3 and global_context_latents.shape[1] == 1:
                    global_context_latents = global_context_latents.squeeze(1)

            # Sinkhorn-Knopp 정규화
            global_context_latents = self.sinkhorn_knopp(
                global_context_latents, teacher_temp=self.teacher_temp
            )

            student_latents = self.proj_distill_student(feats[:, 0])
            student_latents = F.log_softmax(student_latents / self.student_temp, dim=1)
            loss_distill = (
                F.kl_div(
                    student_latents,
                    global_context_latents,
                    reduction="batchmean",
                    log_target=False,
                )
                * self.student_temp
                * self.teacher_temp
            )
            loss_dict["distill"] = (
                self.lambda_distill * loss_distill / self.hparams.gradient_accumulation_steps
            )

        return x, loss_dict

    def _enable_grads(self, module):
        if module:
            for p in module.parameters():
                p.requires_grad = True

    def shared_step(self, batch: Dict[str, Any], batch_idx: int, split="train") -> torch.Tensor:
        """Shared step for training, validation and testing."""
        do_distill = self.lambda_distill > 0
        do_hmi = self.lambda_hmi > 0 and (self.current_epoch >= self.high_freq_start_epoch - 1)
        x, loss_dict = self.compute_loss(batch, split, do_distill, do_hmi)

        # Logging
        self.log_dict(
            {f"{split}_{name}_loss": v for name, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=(split == "train"),
            sync_dist=True,
        )

        loss = sum(loss_dict.values())

        if split == "train":
            opt_main = self._optimizer_main
            opt_distill = self._optimizer_distill
            opt_hmi = self._optimizer_hmi

            self._update_scheduler(opt_main, "main")
            if do_distill and opt_distill:
                self._update_scheduler(opt_distill, "distill")
            if do_hmi and opt_hmi:
                self._update_scheduler(opt_hmi, "hmi")

            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
                # Main
                if self.hparams.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), self.hparams.gradient_clip_val, norm_type=2
                    )
                self._log_grad_norm(self.encoder.parameters(), split="train", prefix="encoder")
                opt_main.step()
                opt_main.zero_grad(set_to_none=True)

                # Distill
                if do_distill and opt_distill:
                    if self.hparams.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(
                            self.proj_distill_student.parameters(),
                            self.hparams.gradient_clip_val,
                            norm_type=2,
                        )
                    self._log_grad_norm(
                        self.proj_distill_student.parameters(),
                        split="train",
                        prefix="proj_distill_student",
                    )
                    opt_distill.step()
                    opt_distill.zero_grad(set_to_none=True)

                # HMI
                if do_hmi and opt_hmi:
                    if self.hparams.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(
                            self.mi_max_high_freq.parameters(),
                            self.hparams.gradient_clip_val,
                            norm_type=2,
                        )
                    self._log_grad_norm(
                        self.mi_max_high_freq.parameters(),
                        split="train",
                        prefix="mi_max_high_freq",
                    )
                    opt_hmi.step()
                    opt_hmi.zero_grad(set_to_none=True)

                # EMA
                if hasattr(self, "ema") and do_hmi:
                    decay = self.momentum_schedule[self.step_count]
                    self.ema.update(self.encoder.backbone, decay=decay)
                    self.log(
                        "train/ema_high_freq_momentum",
                        decay,
                        sync_dist=True,
                        rank_zero_only=True,
                        on_step=True,
                        on_epoch=False,
                    )

            if (self.train_data_to_visualize is None) and ((self.current_epoch + 1) % 10 == 0):
                self.train_data_to_visualize = x

            self.step_count += 1

        elif split == "val":
            # 배치에서 직접 이미지를 가져오기
            self.val_data_to_visualize["img"].append(batch["img"])

        return loss.detach()

    def _update_scheduler(self, optimizer, name):
        """Helper to update LR and Weight Decay."""
        if optimizer is None:
            return

        if name in self.wd_schedules and self.wd_schedules[name] is not None:
            wd = self.wd_schedules[name][self.step_count]
            self._set_weight_decay(optimizer, wd)
            tag = "train/weight_decay" if name == "main" else f"train/weight_decay_{name}"
            self.log(tag, wd, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False)

        if name in self.lr_schedules and self.lr_schedules[name] is not None:
            lr = self.lr_schedules[name][self.step_count]
            self._set_lr(optimizer, lr)
            tag = "train/lr" if name == "main" else f"train/lr_{name}"
            self.log(tag, lr, sync_dist=True, rank_zero_only=True, on_step=True, on_epoch=False)

    def _set_weight_decay(self, opt: torch.optim.Optimizer, wd: float):
        for i, g in enumerate(opt.param_groups):
            if i == 0:
                g["weight_decay"] = wd
                break

    def _set_lr(self, opt: torch.optim.Optimizer, lr: float):
        opt.param_groups[0]["lr"] = lr

    def sinkhorn_knopp(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float = 0.07,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """Sinkhorn-Knopp Normalization."""
        logits = (teacher_output / teacher_temp).float()
        logits_t = logits.t()
        logits_max = logits_t.max(dim=1, keepdim=True)[0]
        logits_stable = logits_t - logits_max
        Q = logits_stable.exp()
        if torch.isnan(Q).any():
            raise ValueError("NaN detected in Q after exp().")

        K, B_local = Q.shape
        world_size = self.trainer.world_size if getattr(self, "trainer", None) is not None else 1
        B = B_local * world_size

        def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x

        sum_Q = _all_reduce_sum(Q.sum())
        Q /= sum_Q

        for it in range(n_iterations):
            sum_of_rows = _all_reduce_sum(Q.sum(dim=1, keepdim=True))
            Q /= sum_of_rows
            Q /= K

            sum_of_cols = Q.sum(dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B
        return Q.t().contiguous()

    @torch.inference_mode()
    @rank_zero_only
    def _visualize_masks(self):
        images = self.val_data_to_visualize["img"]
        if not images:
            return
        images = torch.cat(images, dim=0)[torch.randperm(len(images))[:4]]

        res = self.forward_ssl_encoder(images, masking=True)
        masks_mim = (
            einops.rearrange(
                res["mask_mim"],
                "b (h w) -> b h w",
                h=images.shape[2] // self.encoder.patch_size,
            )
            .repeat_interleave(self.encoder.patch_size, dim=-1)
            .repeat_interleave(self.encoder.patch_size, dim=-2)
        )[:, None, :, :]
        visible_target_images = torch.where(
            masks_mim == 0.5,
            images,
            torch.where(
                masks_mim == 1.0,
                0.5 * images
                + 0.5 * torch.tensor([0.0, 1.0, 0.0], device=images.device).view(1, 3, 1, 1),
                0,
            ),
        )

        concat = torch.cat(
            [
                torch.cat(
                    [
                        images[b],
                        visible_target_images[b],
                    ],
                    dim=-1,
                )
                for b in range(images.shape[0])
            ],
            dim=-2,
        )

        self.logging_util.log_image_tensor(self, concat.to("cpu"), "masks_mim")

    def on_validation_start(self):
        """Method called when the validation phase begins.

        Initializes the counter for each target class to 0.

        Returns:
            None
        """
        self.class_counters = {cls: 0 for cls in self.target_class_list}

    def on_train_epoch_end(self):
        """Called at the end of train epoch."""
        self._visualize_masked_images(split="train")

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch.

        Performs visualization and logging of DINO reconstructions and attention maps.
        """
        self._visualize_patch_feature_norm()
        self._visualize_patch_pca_features()
        self._visualize_masked_images()

    @torch.inference_mode()
    @rank_zero_only
    def _visualize_masked_images(self, split="val"):
        """Visualizes masked images."""
        self.encoder.eval()
        self.decoder.eval()

        # Prepare image data
        if split == "val":
            images = self.val_data_to_visualize["img"]
            images = torch.cat(images, dim=0)[torch.randperm(len(images))[:3]]
            res = self.forward_ssl_encoder(images, masking=True, fixed_mask=True)
        else:
            if self.train_data_to_visualize is None:
                return

            images = self.train_data_to_visualize[:3]
            res = self.forward_ssl_encoder(images, masking=True)

        masks = (
            einops.rearrange(
                res["mask_mim"],
                "b (h w) -> b h w",
                h=images.shape[2] // self.encoder.patch_size,
            )
            .repeat_interleave(self.encoder.patch_size, dim=1)
            .repeat_interleave(self.encoder.patch_size, dim=2)
        )[:, None, :, :]
        preds = self.forward_decoder(res["features"], res["restore_ids_mim"])["patch"]
        # unormalize
        patches = patchify(images, self.encoder.patch_size)
        means = patches.mean(dim=-1, keepdim=True)
        vars = patches.var(dim=-1, keepdim=True)
        preds = ((preds * torch.sqrt(vars + 1e-6)) + means).clamp(0, 1)
        # unpatchify
        preds = unpatchify(preds, self.encoder.patch_size)

        masked_images = images.clone()
        # In mask patches, 0.5 indicates prediction tokens
        # In mask patches, 1 indicates reconstruction tokens
        # In mask patches, 0 indicates ignored tokens
        dark_gray = (96 / 255, 96 / 255, 96 / 255)  # rgb(96, 96, 96)
        dark_gray = torch.tensor(dark_gray).view(1, 3, 1, 1).to(masks.device)
        masked_images = torch.where(
            masks == 0,
            0,
            torch.where(masks == 1, dark_gray, masked_images),
        )
        recon_masked_patches_only = torch.where(
            masks == 1,
            preds,
            torch.where(masks == 0.5, images * 0.5 + 0.5 * dark_gray, 0),
        )
        recon_all_patches = torch.where(masks == 1, preds, images)

        # Margin and image concatenation for visualization
        concat = torch.cat(
            [
                torch.cat(
                    [
                        images[b],
                        masked_images[b],
                        recon_masked_patches_only[b],
                        recon_all_patches[b],
                    ],
                    dim=-1,
                )
                for b in range(images.shape[0])
            ],
            dim=-2,
        )

        self.logging_util.log_image_tensor(self, concat.to("cpu"), f"masked_images/{split}")

        if split == "train":
            self.encoder.train()
            self.decoder.train()


class ModelEMA:
    def __init__(self, model: torch.nn.Module):
        # Create a copy of the model for accumulating the EMA weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()  # EMA model is only for inference
        for p in self.ema_model.parameters():
            p.requires_grad_(False)  # no gradients for EMA model

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: float = 0.999):
        # Update each EMA parameter: shadow = decay * shadow + (1−decay) * param
        # Convert decay to Python float if it's a tensor or numpy type
        if isinstance(decay, torch.Tensor):
            decay = decay.item()
        elif isinstance(decay, (np.floating, np.integer)):
            decay = float(decay)
        decay = float(decay)
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            # Ensure model_v is on the same device as ema_v
            if model_v.device != ema_v.device:
                model_v = model_v.to(ema_v.device)
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)
