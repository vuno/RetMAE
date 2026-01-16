#!/usr/bin/env python3
"""
cka_dump_meta_single.py

- Generates only CKA meta data for a SINGLE model (default: FunMMA) — no plots
- Computes diagonal CKA between HF/LF and Full (within the SAME model), optionally saves full matrix
- Saves epoch-wise late-layer mean to CSV
- Does NOT recompute MAE; use this when only the target model is updated
- Results are written into a fresh timestamped run directory (no overwrite)

Memory safety:
- context manager-based clean-up, hook removal, GPU/CPU transfer, dereferencing, empty_cache()

Usage (example):
    python cka_dump_meta_single.py \\
      --model_ckpt_99 /path/to/epoch_099.ckpt \\
      --save_dir /path/to/cka_res \\
      --epoch_points 19,49,99 \\
      --snapshot_epochs 19,49,99 \\
      --model_name FunMMA --prefix fun \\
      --save_mats_all
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==== Project imports ====
sys.path.append("your_path")
from src.common.data.base import Dataframe
from src.tasks.pretrain.encoder import SSLEncoder

torch.manual_seed(123)
np.random.seed(123)


# ---------- Utilities ----------
def infer_epoch_width(ckpt_path: str) -> Tuple[int, re.Pattern]:
    """
    Infers the width of the epoch number in the checkpoint path and compiles a regex pattern to match it.
    Supports checkpoint paths like 'epoch_099' or 'epoch=099'.

    Args:
        ckpt_path (str): The checkpoint path containing the epoch number.

    Returns:
        Tuple[int, re.Pattern]: The width of the epoch number and a compiled regex pattern capturing the prefix as group(1).
    """
    m = re.search(r"(epoch[_=])(\d+)", ckpt_path)
    if not m:
        raise ValueError(
            "Cannot infer epoch pattern from ckpt path (expected 'epoch_\\d+' or 'epoch=\\d+')."
        )
    width = len(m.group(2))
    # Correct quantifier: \d{width}
    pattern = re.compile(rf"(epoch[_=])\d{{{width}}}")
    return width, pattern


def make_epoch_paths(ckpt_template: str, epoch_points: List[int]) -> Dict[int, str]:
    """
    Creates a mapping from epoch number to checkpoint path by substituting the epoch digits in a template path.

    Args:
        ckpt_template (str): The template checkpoint path containing the epoch number.
        epoch_points (List[int]): List of epoch numbers.

    Returns:
        Dict[int, str]: Mapping from epoch number to checkpoint path.
    """
    width, patt = infer_epoch_width(ckpt_template)
    d = {}
    for e in epoch_points:
        repl = rf"\g<1>{str(e).zfill(width)}"  # keep 'epoch_' or 'epoch=' prefix
        new_path, n = patt.subn(repl, ckpt_template, count=1)
        if n == 0:
            raise ValueError(
                f"[make_epoch_paths] Failed to substitute epoch in path: {ckpt_template}"
            )
        d[e] = new_path
    return d


def get_backbone(weights_path: str) -> nn.Module:
    """
    Loads the backbone model from the given weights path using the SSLEncoder wrapper.

    Args:
        weights_path (str): Path to the model checkpoint.

    Returns:
        nn.Module: The loaded backbone model.
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    encoder = SSLEncoder(
        model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m",
        algorithm="FmiMAE",
        pretrained=True,
        ckpt_path=weights_path,
        mask_ratio=0.0,
        img_size=224,
        patch_size=14,
        reg_tokens=4,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        standardize=True,
        dynamic_img_size=True,
        norm_layer=nn.LayerNorm,
    )
    return encoder.backbone


class ImageDataset(Dataset):
    """
    Dataset for loading images from file paths specified in a DataFrame.

    The DataFrame must contain a column named 'filepath' with the image file paths.
    Images are resized to 224x224 and converted to tensors.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image file paths in the 'filepath' column.
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        image_path = self.data.iloc[idx]["filepath"]
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)
        if x.max() > 1:
            x = x / 255.0
        return x


def build_dataframe() -> pd.DataFrame:
    """
    Concatenates multiple public fundus datasets into a single DataFrame with a 'filepath' column.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all image file paths.
    """
    dataframes = {
        "RFMiD": Dataframe(
            root="your_path/RFMiD2",
            image_folder_name="images",
            label_file_name="labels-duplicate_rm",
            suffix="base_proc-224x224",
        ).dataframe,
        "IDRiD": Dataframe(
            root="your_path/IDRiD",
            image_folder_name="images",
            label_file_name="labels-duplicate_rm",
            suffix="base_proc-224x224",
        ).dataframe,
        "CHAKSU": Dataframe(
            root="your_path/CHAKSU",
            image_folder_name="images",
            label_file_name="labels-duplicate_rm",
            suffix="base_proc-224x224",
        ).dataframe,
        "APTOS": Dataframe(
            root="your_path/Drishti-GS1",
            image_folder_name="images",
            label_file_name="labels-duplicate_rm",
            suffix="base_proc-224x224",
        ).dataframe,
        "ADAM": Dataframe(
            root="your_path/ADAM",
            image_folder_name="images",
            label_file_name="labels-duplicate_rm",
            suffix="base_proc-224x224",
        ).dataframe,
    }
    return pd.concat(list(dataframes.values()), ignore_index=True)


# ---------- CKA ----------
class CKA_FunMMA:
    """
    Context manager for CKA (Centered Kernel Alignment) computation between two (cloned) models.

    This class:
    - Registers forward hooks on specified layers of both models to capture features.
    - Accumulates HSIC (Hilbert-Schmidt Independence Criterion) constituents across batches.
    - Performs cosine-CKA normalization after accumulation.
    - Ensures clean shutdown: removes hooks, clears feature dictionaries, moves models to CPU, dereferences, and empties CUDA cache.
    """

    def __init__(
        self,
        model1,
        model2,
        model1_name=None,
        model2_name=None,
        model1_layers=None,
        model2_layers=None,
        device="cpu",
    ):
        """
        Args:
            model1 (nn.Module): The first model (e.g., full model).
            model2 (nn.Module): The second model (e.g., HF/LF variant).
            model1_name (str, optional): Name for the first model.
            model2_name (str, optional): Name for the second model.
            model1_layers (list, optional): List of layer names to hook in model1.
            model2_layers (list, optional): List of layer names to hook in model2.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.model1 = model1.eval().to(device)
        self.model2 = model2.eval().to(device)
        self.device = device

        self.model1_info, self.model2_info = {}, {}
        self.model1_info["Name"] = model1_name or model1.__class__.__name__
        self.model2_info["Name"] = model2_name or model2.__class__.__name__
        if self.model1_info["Name"] == self.model2_info["Name"]:
            warn(f"Both models named '{self.model1_info['Name']}'. Consider unique names.")

        self.model1_layers = model1_layers
        self.model2_layers = model2_layers
        self._allowed1 = set(model1_layers) if model1_layers is not None else None
        self._allowed2 = set(model2_layers) if model2_layers is not None else None

        self.model1_info["Layers"] = []
        self.model2_info["Layers"] = []

        self.model1_features: Dict[str, torch.Tensor] = {}
        self.model2_features: Dict[str, torch.Tensor] = {}
        self._hook_handles = []

        self.hsic_matrix: torch.Tensor = None  # set in compare
        self._insert_hooks()

    def _log_layer(self, model_key, layer_name, module, inp, out):
        """
        Internal hook function to log the output of a layer during forward pass.

        Args:
            model_key (str): 'model1' or 'model2'.
            layer_name (str): Name of the layer.
            module (nn.Module): The layer module.
            inp: Input to the layer.
            out: Output from the layer.
        """
        if model_key == "model1":
            self.model1_features[layer_name] = out
        else:
            self.model2_features[layer_name] = out

    def _insert_hooks(self):
        """
        Registers forward hooks on the specified layers of both models.
        """
        for name, layer in self.model1.named_modules():
            if self._allowed1 is None or name in self._allowed1:
                self.model1_info["Layers"].append(name)
                h = layer.register_forward_hook(partial(self._log_layer, "model1", name))
                self._hook_handles.append(h)
        for name, layer in self.model2.named_modules():
            if self._allowed2 is None or name in self._allowed2:
                self.model2_info["Layers"].append(name)
                h = layer.register_forward_hook(partial(self._log_layer, "model2", name))
                self._hook_handles.append(h)

    def remove_hooks(self):
        """
        Removes all registered forward hooks from both models.
        """
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception as e:
                raise e
        self._hook_handles.clear()

    def close(self):
        """
        Cleans up resources: removes hooks, clears feature dicts, moves models to CPU, dereferences, and empties CUDA cache.
        """
        self.remove_hooks()
        self.model1_features.clear()
        self.model2_features.clear()
        if isinstance(self.hsic_matrix, torch.Tensor) and self.hsic_matrix.is_cuda:
            self.hsic_matrix = self.hsic_matrix.cpu()
        try:
            if self.model1 is not None:
                self.model1.to("cpu")
            if self.model2 is not None:
                self.model2.to("cpu")
        except Exception as e:
            raise e
        self.model1 = None
        self.model2 = None
        torch.cuda.empty_cache()

    def __del__(self):
        """
        Destructor to ensure resources are cleaned up.
        """
        try:
            self.close()
        except Exception as e:
            raise e

    def __enter__(self):
        """
        Enters the context manager.

        Returns:
            CKA_FunMMA: The context manager instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Exits the context manager and cleans up resources.
        """
        self.close()

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """
        Centers a Gram matrix.

        Args:
            K (torch.Tensor): The Gram matrix to center.

        Returns:
            torch.Tensor: The centered Gram matrix.
        """
        n = K.size(0)
        unit = torch.ones(n, n, device=K.device) / n
        H = torch.eye(n, device=K.device) - unit
        return H @ K @ H

    def _HSIC(self, K, L):
        """
        Computes the Hilbert-Schmidt Independence Criterion (HSIC) between two Gram matrices.

        Args:
            K (torch.Tensor): First Gram matrix.
            L (torch.Tensor): Second Gram matrix.

        Returns:
            torch.Tensor: The HSIC value.
        """
        Kc = self._center_gram(K)
        Lc = self._center_gram(L)
        hsic_val = (Kc * Lc).sum()
        n = K.shape[0]
        return hsic_val / ((n - 1) ** 2)

    @torch.inference_mode()
    def compare(
        self,
        dataloader,
        infer_high_freq=False,
        infer_low_freq=False,
        random_masking=False,
        mask_ratio=0.0,
        max_batches=None,
    ):
        """
        Compares the representations of two models using CKA across the specified layers.

        Args:
            dataloader (DataLoader): DataLoader providing input images.
            infer_high_freq (bool): Whether to use high-frequency inference for model2.
            infer_low_freq (bool): Whether to use low-frequency inference for model2.
            random_masking (bool): Whether to use random masking for model2.
            mask_ratio (float): Masking ratio for random masking.
            max_batches (int, optional): Maximum number of batches to process.
        """
        assert (
            infer_high_freq or infer_low_freq or random_masking
        ), "Either infer_high_freq or infer_low_freq or random_masking must be True"

        N = len(self.model1_info["Layers"])
        M = len(self.model2_info["Layers"])
        self.hsic_matrix = torch.zeros(N, M, 3, device=self.device)

        num_batches = len(dataloader)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        it = tqdm(dataloader, total=num_batches, desc="| Comparing features |", leave=False)
        for bi, x in enumerate(it):
            if bi >= num_batches:
                break
            self.model1_features.clear()
            self.model2_features.clear()

            x = x.to(self.device, non_blocking=True)

            _ = self.model1.forward_features(x)
            _ = self.model2.forward_features(
                x,
                infer_high_freq=infer_high_freq,
                infer_low_freq=infer_low_freq,
                masking=True if random_masking else False,
                random_masking=random_masking,
                mask_ratio=mask_ratio,
            )

            for i, (_, feat1) in enumerate(self.model1_features.items()):
                X = feat1[:, :1].flatten(1)  # CLS only
                K = X @ X.t()
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (_, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2[:, :1].flatten(1)
                    L = Y @ Y.t()
                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

            # release
            del x, feat1, feat2, K, L
            torch.cuda.empty_cache()

        # cosine-CKA normalization
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (
            self.hsic_matrix[:, :, 0].clamp(min=1e-12).sqrt()
            * self.hsic_matrix[:, :, 2].clamp(min=1e-12).sqrt()
        )
        self.hsic_matrix = self.hsic_matrix.to("cpu")
        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NaNs"
        torch.cuda.empty_cache()

    def export_matrix(self) -> np.ndarray:
        """
        Exports the computed CKA matrix as a NumPy array.

        Returns:
            np.ndarray: The CKA matrix.
        """
        return self.hsic_matrix.numpy()


def compute_cka_diag_and_matrix(
    backbone: nn.Module, dataloader, layer_names, device, band, max_batches=None
):
    """
    Computes the diagonal and full CKA matrix between the full model and its HF or LF variant.

    Args:
        backbone (nn.Module): The backbone model.
        dataloader (DataLoader): DataLoader providing input images.
        layer_names (list): List of layer names to compare.
        device (str): Device to use ('cpu' or 'cuda').
        band (str): 'hf' for high-frequency, 'lf' for low-frequency.
        max_batches (int, optional): Maximum number of batches to process.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The diagonal and full CKA matrix.
    """
    assert band in {"hf", "lf"}
    with CKA_FunMMA(
        model1=backbone,
        model2=deepcopy(backbone),
        model1_name="Full",
        model2_name=("HF" if band == "hf" else "LF"),
        model1_layers=layer_names,
        model2_layers=layer_names,
        device=device,
    ) as cka:
        cka.compare(
            dataloader,
            infer_high_freq=(band == "hf"),
            infer_low_freq=(band == "lf"),
            max_batches=max_batches,
        )
        M = cka.export_matrix()
    diag = np.diag(M)
    return diag, M


def main():
    """
    Main entry point for dumping CKA meta data for a single model.

    This function:
    - Parses command-line arguments.
    - Loads data and model checkpoints for specified epochs.
    - Computes CKA diagonals and matrices for HF and LF variants.
    - Saves results (CSV, matrices, meta) in a timestamped run directory.
    """
    ap = argparse.ArgumentParser("Dump CKA meta for a SINGLE model (no plotting)")
    ap.add_argument(
        "--model_ckpt_99",
        required=True,
        help="Template ckpt path (contains 'epoch_XXX' or 'epoch=XXX').",
    )
    ap.add_argument("--model_name", default="FunMMA", help="Model display name (for meta).")
    ap.add_argument(
        "--prefix", default="fun", help="Prefix for CSV columns and file stems (e.g., 'fun')."
    )
    ap.add_argument(
        "--save_dir",
        default="your_path/experiment/cka_res",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--epoch_points", default="9,19,29,39,49,59,69,79,89,99")
    ap.add_argument("--snapshot_epochs", default="19,49,99")
    ap.add_argument("--late_k", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument(
        "--save_mats_all",
        action="store_true",
        help="Save full CKA matrices for all epochs (default: only snapshot epochs).",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Timestamped run dir (no overwrite)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.save_dir)
    run_dir = base_dir / f"run_{ts}"
    (run_dir / "csv").mkdir(parents=True, exist_ok=True)
    (run_dir / "mats").mkdir(parents=True, exist_ok=True)

    # Data
    df = build_dataframe()
    loader = DataLoader(
        ImageDataset(df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Layers
    layer_num = 12
    layer_names = [f"model.blocks.{i}" for i in range(layer_num)]
    layer_idxs = list(range(layer_num))

    # Epochs
    epoch_points = [int(x) for x in args.epoch_points.split(",") if x]
    snapshot_epochs = {int(x) for x in args.snapshot_epochs.split(",") if x}

    # Resolve ckpt paths
    ckpts = make_epoch_paths(args.model_ckpt_99, epoch_points)
    # Sanity check
    for e in epoch_points:
        p = ckpts[e]
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing ckpt for epoch {e}: {p}")
    # Basic diversity check
    if len(set(ckpts.values())) == 1:
        warn("[warn] All epoch paths are identical. Check your ckpt template pattern.")

    # Storage for epoch-wise means
    late_series = {args.prefix.upper(): {"hf": [], "lf": []}}

    for e in epoch_points:
        print(f"\n=== Epoch {e} ===")
        bb = get_backbone(ckpts[e])
        bb.to(device, non_blocking=True)

        diag_hf, mat_hf = compute_cka_diag_and_matrix(
            bb, loader, layer_names, device, "hf", args.max_batches
        )
        torch.cuda.empty_cache()

        diag_lf, mat_lf = compute_cka_diag_and_matrix(
            bb, loader, layer_names, device, "lf", args.max_batches
        )

        # Release model
        bb.to("cpu")
        del bb
        torch.cuda.empty_cache()

        # Save per-epoch diagonals
        pd.DataFrame(
            {
                "layer": layer_idxs,
                f"{args.prefix}_hf": diag_hf,
                f"{args.prefix}_lf": diag_lf,
            }
        ).to_csv(run_dir / "csv" / f"diag_epoch{e:03d}.csv", index=False)

        # Save matrices only for snapshots or all
        if args.save_mats_all or (e in snapshot_epochs):
            np.savez_compressed(
                run_dir / "mats" / f"epoch{e:03d}_{args.prefix}_hf.npz",
                M=mat_hf.astype(np.float32),
            )
            np.savez_compressed(
                run_dir / "mats" / f"epoch{e:03d}_{args.prefix}_lf.npz",
                M=mat_lf.astype(np.float32),
            )

        # Late-layer mean (last k)
        k = max(1, min(args.late_k, len(diag_hf)))
        late_series[args.prefix.upper()]["hf"].append(float(diag_hf[-k:].mean()))
        late_series[args.prefix.upper()]["lf"].append(float(diag_lf[-k:].mean()))

        # Cleanup
        del diag_hf, mat_hf, diag_lf, mat_lf
        torch.cuda.empty_cache()

    # Save epoch-wise trajectory
    pd.DataFrame(
        {
            "epoch": epoch_points,
            f"{args.prefix}_hf": late_series[args.prefix.upper()]["hf"],
            f"{args.prefix}_lf": late_series[args.prefix.upper()]["lf"],
        }
    ).to_csv(run_dir / "csv" / "epochwise_trajectory.csv", index=False)

    # Save meta
    meta = {
        "epoch_points": epoch_points,
        "snapshot_epochs": sorted(list(snapshot_epochs)),
        "late_k": int(args.late_k),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_batches": (None if args.max_batches is None else int(args.max_batches)),
        "model_ckpt_99": args.model_ckpt_99,
        "model_name": args.model_name,
        "prefix": args.prefix,
        "layer_num": layer_num,
        "bands": ["hf", "lf"],
        "paths": {
            "root": str(run_dir.resolve()),
            "csv": str((run_dir / "csv").resolve()),
            "mats": str((run_dir / "mats").resolve()),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: write/update a 'latest' symlink for convenience
    try:
        latest = base_dir / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir, target_is_directory=True)
    except Exception as e:
        warn(f"[note] Could not create 'latest' symlink: {e}")

    print(f"\n[OK] Meta dumped at: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
