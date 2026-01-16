from pathlib import Path
from typing import Dict, Optional

import rootutils
import torch
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SSLEncoder(nn.Module):
    """Self-supervised learning encoder class.

    This class implements a unified encoder interface for self-supervised learning tasks,
    specifically optimized for retinal fundus image analysis. It supports various backbone
    models including RetMAE, providing consistent API for feature extraction, attention
    visualization, and intermediate feature access.

    Args:
        model_name (str): Name of the model to load (e.g., 'vit_base_patch16_224').
        algorithm (str, optional): Self-supervised learning algorithm to use.
            Must be one of ["RetMAE", "TimmViTBackbone", "FLAIR", "KeepFIT",
            "UrFound", "RETClipBackbone", "RETFoundBackbone"]. Defaults to "RetMAE".
        ckpt_path (Optional[str], optional): Path to checkpoint file to load. Defaults to None.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
        mask_ratio (float, optional): Ratio of patches to mask during training.
        visible_ratio (Optional[float], optional): Ratio of visible patches for masking.
        random_masking (Optional[bool], optional): Whether to use random masking strategy.
        fixed_mask (Optional[bool], optional): Whether to use fixed masking for reproducibility.
        infer_high_freq (bool, optional): Whether to infer high-frequency features.
        infer_low_freq (bool, optional): Whether to infer low-frequency features.
        infer_random (bool, optional): Whether to infer random features.
        load_ckpt_strict (bool, optional): Whether to load checkpoint strictly. Defaults to True.
        **model_params: Additional model parameters passed to the backbone.

    Attributes:
        backbone: The underlying backbone model (RetMAE, TimmViTBackbone, etc.)
        patch_size (int): Size of image patches used by the model
        mask_ratio (Optional[float]): Ratio of patches to mask during training
        visible_ratio (Optional[float]): Ratio of visible patches for masking
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        algorithm: str = "RetMAE",
        ckpt_path: Optional[str] = None,
        pretrained: bool = False,
        mask_ratio: Optional[float] = None,
        # FmiMAEv2
        visible_ratio: Optional[float] = None,
        random_masking: Optional[bool] = None,
        fixed_mask: Optional[bool] = None,
        infer_high_freq: bool = False,
        infer_low_freq: bool = False,
        infer_random: bool = False,
        load_ckpt_strict: bool = True,
        **model_params,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.visible_ratio = visible_ratio
        # FmiMAE
        self.random_masking = random_masking
        self.fixed_mask = fixed_mask
        self.infer_high_freq = infer_high_freq
        self.infer_low_freq = infer_low_freq
        self.infer_random = infer_random

        # Validate checkpoint path if provided
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

        # Validate algorithm
        valid_algorithms = [
            "RetMAE",
            "TimmViTBackbone",
            "FLAIR",
            "KeepFIT",
            "UrFound",
            "RETClipBackbone",
            "RETFoundBackbone",
        ]
        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
        # Initialize backbone based on algorithm
        try:
            if algorithm == "RetMAE":
                from src.tasks.pretrain.backbone import RetMAE

                self.backbone = RetMAE(
                    model_name=model_name,
                    pretrained=pretrained,
                    ckpt_path=ckpt_path,
                    load_ckpt_strict=load_ckpt_strict,
                    **model_params,
                )

            elif algorithm == "TimmViTBackbone":
                from src.tasks.pretrain.backbone import TimmViTBackbone

                self.backbone = TimmViTBackbone(
                    model_name=model_name,
                    pretrained=pretrained,
                    ckpt_path=ckpt_path,
                    **model_params,
                )

            elif algorithm == "FLAIR":
                from src.tasks.pretrain.backbone import FLAIR

                self.backbone = FLAIR(**model_params)

            elif algorithm == "KeepFIT":
                from src.tasks.pretrain.backbone import KeepFIT

                self.backbone = KeepFIT(**model_params)

            elif algorithm == "UrFound":
                from src.tasks.pretrain.backbone import UrFound

                self.backbone = UrFound(**model_params)

            elif algorithm == "RETClipBackbone":
                from src.tasks.pretrain.backbone import RETClipBackbone

                self.backbone = RETClipBackbone(**model_params)

            elif algorithm == "RETFoundBackbone":
                from src.tasks.pretrain.backbone import RETFoundBackbone

                self.backbone = RETFoundBackbone(**model_params)

            elif algorithm == "VilRef":
                from src.tasks.pretrain.backbone import VilRef

                self.backbone = VilRef(**model_params)

            elif algorithm == "PreTI":
                from src.tasks.pretrain.backbone import PreTI

                self.backbone = PreTI(**model_params)

        except ImportError as e:
            raise ImportError(f"Failed to import backbone for algorithm {algorithm}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {algorithm} backbone: {e}")

        # Ensure backbone has required attributes
        if not hasattr(self.backbone, "patch_size"):
            raise AttributeError(f"Backbone {algorithm} must have 'patch_size' attribute")

        self.patch_size = (
            self.backbone.patch_size
            if not isinstance(self.backbone.patch_size, tuple)
            else self.backbone.patch_size[0]
        )

        self.embed_dim = model_params.get("embed_dim", None)
        if self.embed_dim is None:
            raise ValueError("embed_dim is not set")

        # Set additional attributes from model_params
        for k, v in model_params.items():
            if not hasattr(self, k):  # Avoid overriding existing attributes
                setattr(self, k, v)

    def forward(self, x: torch.Tensor, masking: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size,
                C is number of channels (typically 3 for RGB), H is height, W is width.
            masking (bool, optional): Whether to apply masking during forward pass.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to forward_features.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - features: Encoded feature representations of shape (B, N, D)
                - mask_mim: Binary mask indicating which patches were masked
                - restore_ids_mim: Indices to restore the original patch order (if available)
                - high_freq: High-frequency features (if available)

        Raises:
            RuntimeError: If forward pass fails due to model issues.
        """
        try:
            res = self.forward_features(
                x=x,
                mask_ratio=self.mask_ratio,
                masking=masking,
                random_masking=self.random_masking,
                visible_ratio=self.visible_ratio,
                infer_random=self.infer_random,
                infer_high_freq=self.infer_high_freq,
                infer_low_freq=self.infer_low_freq,
                **kwargs,
            )

            # Validate required keys in response
            if "features" not in res:
                raise KeyError("Backbone forward_features must return 'features' key")
            if "mask_mim" not in res:
                raise KeyError("Backbone forward_features must return 'mask_mim' key")

            return {
                "features": res["features"],
                "mask_mim": res["mask_mim"],
                "restore_ids_mim": res.get("restore_ids_mim", None),
                "high_freq": res.get("high_freq", None),
            }
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}") from e

    def forward_features(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        fixed_mask: bool = False,
        masking: Optional[bool] = None,
        random_masking: Optional[bool] = None,
        visible_ratio: Optional[float] = None,
        infer_random: bool = False,
        infer_high_freq: bool = False,
        infer_low_freq: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract features from the backbone model.

        This method delegates to the backbone's forward_features method with all
        necessary parameters for RetMAE and other supported algorithms.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            mask_ratio (Optional[float], optional): Ratio of patches to mask.
                Defaults to None (uses instance default).
            fixed_mask (bool, optional): Whether to use fixed mask for reproducibility.
                Defaults to False.
            masking (Optional[bool], optional): Whether to apply masking strategy.
                Defaults to None (uses instance default).
            random_masking (Optional[bool], optional): Whether to use random masking.
                Defaults to None (uses instance default).
            visible_ratio (Optional[float], optional): Ratio of visible patches.
                Defaults to None.
            infer_random (bool, optional): Whether to infer random features. Defaults to False.
            infer_high_freq (bool, optional): Whether to infer high-frequency features.
                Defaults to False.
            infer_low_freq (bool, optional): Whether to infer low-frequency features.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing extracted features and masks.
                Required keys: 'features', 'mask_mim'
                Optional keys: 'restore_ids_mim', 'high_freq'

        Raises:
            AttributeError: If backbone does not have forward_features method.
            RuntimeError: If feature extraction fails.
        """
        if not hasattr(self.backbone, "forward_features"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} does not have forward_features method"
            )

        try:
            return self.backbone.forward_features(
                x=x,
                mask_ratio=mask_ratio,
                fixed_mask=fixed_mask,
                masking=masking,
                random_masking=random_masking,
                infer_high_freq=infer_high_freq,
                infer_low_freq=infer_low_freq,
                infer_random=infer_random,
                visible_ratio=visible_ratio,
            )
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}") from e

    def get_self_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract self-attention maps from the backbone model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing attention maps for each stage.

        Raises:
            AttributeError: If backbone does not support attention extraction.
            RuntimeError: If attention extraction fails.
        """
        if not hasattr(self.backbone, "get_self_attention"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} does not support attention extraction"
            )

        try:
            return self.backbone.get_self_attention(x)
        except Exception as e:
            raise RuntimeError(f"Attention extraction failed: {e}") from e

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate features from the backbone model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing intermediate features for each stage.

        Raises:
            AttributeError: If backbone does not support intermediate feature extraction.
            RuntimeError: If feature extraction fails.
        """
        if not hasattr(self.backbone, "get_intermediate_features"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} does not support intermediate feature extraction"
            )

        try:
            return self.backbone.get_intermediate_features(x)
        except Exception as e:
            raise RuntimeError(f"Intermediate feature extraction failed: {e}") from e

    def get_cosine_similarity_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract cosine similarity maps from the backbone model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Cosine similarity map tensor.

        Raises:
            AttributeError: If backbone does not support cosine similarity computation.
            RuntimeError: If similarity computation fails.
        """
        if not hasattr(self.backbone, "get_cosine_similarity_map"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} does not support cosine similarity computation"
            )

        try:
            return self.backbone.get_cosine_similarity_map(x)
        except Exception as e:
            raise RuntimeError(f"Cosine similarity computation failed: {e}") from e

    def get_intermediate_features_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all intermediate features from the backbone model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing all intermediate features.

        Raises:
            AttributeError: If backbone does not support all intermediate feature extraction.
            RuntimeError: If feature extraction fails.
        """
        if not hasattr(self.backbone, "get_intermediate_features_all"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} does not support all intermediate feature extraction"
            )

        try:
            return self.backbone.get_intermediate_features_all(x)
        except Exception as e:
            raise RuntimeError(f"All intermediate feature extraction failed: {e}") from e


if __name__ == "__main__":
    ALGORITHM = "RetMAE"
    MASK_RATIO = 0.75
    MODEL_PARAMS = {
        "img_size": 512,
        "patch_size": 32,
        "model_name": "vit_base_patch16_224",
    }

    net = SSLEncoder(
        algorithm=ALGORITHM,
        mask_ratio=MASK_RATIO,
        **MODEL_PARAMS,
    )
    x = torch.randn(1, 3, 512, 512)
    result = net(x)

    print("features", result["features"].shape)
    print("mask_mim", result["mask_mim"].shape)
    if result.get("restore_ids_mim") is not None:
        print("restore_ids_mim", result["restore_ids_mim"].shape)
    print("Done")
