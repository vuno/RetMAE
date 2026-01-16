#!/usr/bin/env python3
"""
cka_plot_from_meta.py

- Generates plots from meta.json, CSV, and (optionally) .npz matrix files produced by cka_dump_meta.py.
- No model inference is performed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COLOR_FUN = "#0D5EA6"
COLOR_MAE = "#E14434"


def plot_layerwise_lines(diag_df: pd.DataFrame, epoch: int, out_dir: Path):
    """
    Plot layer-wise CKA diagonal values for both MAE and FunMMA models for a given epoch.

    Args:
        diag_df (pd.DataFrame): DataFrame containing columns: layer, mae_hf, mae_lf, fun_hf, fun_lf.
        epoch (int): The epoch number for which the plot is generated.
        out_dir (Path): Output directory to save the generated plot.
    """
    layers = diag_df["layer"].to_numpy()
    for band in ["hf", "lf"]:
        plt.figure(figsize=(8, 4))
        plt.plot(
            layers, diag_df[f"mae_{band}"], marker="o", linestyle="-", label="MAE", color=COLOR_MAE
        )
        plt.plot(
            layers,
            diag_df[f"fun_{band}"],
            marker="o",
            linestyle="-",
            label="FunMMA",
            color=COLOR_FUN,
        )
        plt.xlabel("Transformer Block (layer index)")
        plt.ylabel("Diagonal CKA")
        plt.ylim(0.0, 1.02)
        plt.title(f"Layer-wise CKA — Epoch {epoch} — {'HF' if band=='hf' else 'LF'} token vs Full")
        plt.grid(alpha=0.25)
        plt.legend()
        out = out_dir / f"layerwise_lines_epoch{epoch:03d}_{band}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()


def plot_heatmap_grid(mats_dir: Path, epoch: int, out_dir: Path):
    """
    Plot a 2x2 grid of CKA heatmaps for a given epoch, if the required .npz matrix files exist.

    Args:
        mats_dir (Path): Directory containing the .npz matrix files.
        epoch (int): The epoch number for which the heatmaps are generated.
        out_dir (Path): Output directory to save the generated heatmap plot.

    Notes:
        Expects the following files to exist in mats_dir:
            - epochXXX_mae_hf.npz
            - epochXXX_mae_lf.npz
            - epochXXX_fun_hf.npz
            - epochXXX_fun_lf.npz
    """
    paths = {
        "mae_hf": mats_dir / f"epoch{epoch:03d}_mae_hf.npz",
        "mae_lf": mats_dir / f"epoch{epoch:03d}_mae_lf.npz",
        "fun_hf": mats_dir / f"epoch{epoch:03d}_fun_hf.npz",
        "fun_lf": mats_dir / f"epoch{epoch:03d}_fun_lf.npz",
    }
    if not all(p.exists() for p in paths.values()):
        print(f"[warn] heatmap mats not found for epoch {epoch}; skip heatmaps.")
        return

    M_mae_hf = np.load(paths["mae_hf"])["M"]
    M_mae_lf = np.load(paths["mae_lf"])["M"]
    M_fun_hf = np.load(paths["fun_hf"])["M"]
    M_fun_lf = np.load(paths["fun_lf"])["M"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    vmin, vmax, cmap = 0.0, 1.0, "magma"

    im = axes[0, 0].imshow(M_mae_hf, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("MAE: HF ~ Full")
    axes[0, 1].imshow(M_fun_hf, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("FunMMA: HF ~ Full")
    axes[1, 0].imshow(M_mae_lf, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("MAE: LF ~ Full")
    axes[1, 1].imshow(M_fun_lf, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("FunMMA: LF ~ Full")

    for ax in axes.ravel():
        ax.set_xlabel("Layer (token-b)")
        ax.set_ylabel("Layer (token-a)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(f"CKA Heatmaps per Epoch {epoch}", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    out = out_dir / f"heatmaps_epoch{epoch:03d}.png"
    plt.savefig(out, dpi=220)
    plt.close()


def plot_epochwise_trajectory(traj_csv: Path, out_dir: Path):
    """
    Plot the epoch-wise trajectory of late-layer mean CKA values for both MAE and FunMMA models.

    Args:
        traj_csv (Path): Path to the CSV file containing epoch-wise trajectory data.
        out_dir (Path): Output directory to save the generated plots.

    Notes:
        The CSV file is expected to have columns: epoch, mae_hf, mae_lf, fun_hf, fun_lf.
    """
    df = pd.read_csv(traj_csv)
    epochs = df["epoch"].to_numpy()

    for band in ["hf", "lf"]:
        plt.figure(figsize=(8, 4))
        plt.plot(
            epochs, df[f"mae_{band}"], marker="o", linestyle="-", label="MAE", color=COLOR_MAE
        )
        plt.plot(
            epochs, df[f"fun_{band}"], marker="o", linestyle="-", label="FunMMA", color=COLOR_FUN
        )
        plt.xlabel("Epoch")
        plt.ylabel("Late-layer mean CKA")
        plt.title(f"Epoch-wise Trajectory — {'HF' if band=='hf' else 'LF'} token vs Full")
        plt.grid(alpha=0.25)
        plt.legend()
        out = out_dir / f"epochwise_trajectory_{band}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()


def main():
    """
    Main entry point for generating CKA plots from dumped meta data.

    This function loads meta.json, determines the relevant CSV and matrix directories,
    and generates:
        1) Layer-wise CKA line plots for snapshot epochs.
        2) (Optionally) 2x2 CKA heatmap grids for snapshot epochs.
        3) Epoch-wise trajectory plots of late-layer mean CKA.

    Command-line Arguments:
        --save_dir: Directory containing meta.json, csv/, and mats/ (default: project cka_res).
        --snapshot_epochs: Comma-separated list of snapshot epochs to plot (overrides meta.json).
        --make_heatmaps: If set, generates 2x2 heatmap plots (requires .npz matrix files).
    """
    ap = argparse.ArgumentParser("Plot CKA from dumped meta")
    ap.add_argument(
        "--save_dir",
        default="your_path/src/tasks/pretrain/experiment/cka_res",
        help="Directory that contains meta.json, csv/, mats/",
    )
    ap.add_argument(
        "--snapshot_epochs",
        default=None,
        help="Override snapshot epochs (comma-separated), else use meta.json.",
    )
    ap.add_argument(
        "--make_heatmaps", action="store_true", help="Draw 2x2 heatmaps (needs mats npz files)."
    )
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    with open(save_dir / "meta.json") as f:
        meta = json.load(f)

    csv_dir = Path(meta["paths"]["csv"])
    mats_dir = Path(meta["paths"]["mats"])

    # 1) Layer-wise snapshot lines
    if args.snapshot_epochs is not None:
        snapshot_epochs = [int(x) for x in args.snapshot_epochs.split(",") if x]
    else:
        snapshot_epochs = meta.get("snapshot_epochs", [])
    snap_out = save_dir / "snapshots"
    snap_out.mkdir(exist_ok=True, parents=True)

    for e in snapshot_epochs:
        diag_csv = csv_dir / f"diag_epoch{e:03d}.csv"
        if not diag_csv.exists():
            print(f"[warn] missing diag csv for epoch {e}: {diag_csv}")
            continue
        diag_df = pd.read_csv(diag_csv)
        plot_layerwise_lines(diag_df, epoch=e, out_dir=snap_out)
        if args.make_heatmaps:
            plot_heatmap_grid(mats_dir, epoch=e, out_dir=(save_dir / "heatmaps"))

    # 2) Epoch-wise trajectories
    plot_epochwise_trajectory(csv_dir / "epochwise_trajectory.csv", out_dir=save_dir)
    print(f"[OK] Plots saved under: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
