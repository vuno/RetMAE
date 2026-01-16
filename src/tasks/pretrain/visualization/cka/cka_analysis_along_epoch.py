import os
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("your_path")
from src.common.data.base import Dataframe
from src.tasks.pretrain.encoder import SSLEncoder

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
WEIGHTS_DIR = "your_path/save/fundus_ssl/logs/ssl-freqmae3-vit_base_224_14-42/runs/mae_0.05/checkpoints"


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class CKA_FreqMAE:
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_name: str = None,
        model2_name: str = None,
        model1_layers: List[str] = None,
        model2_layers: List[str] = None,
        device: str = "cpu",
    ):
        # Save models and names
        self.model1 = model1.eval()
        self.model2 = model2.eval()
        self.device = device

        # Info dicts
        self.model1_info, self.model2_info = {}, {}
        self.model1_info["Name"] = model1_name or model1.__class__.__name__
        self.model2_info["Name"] = model2_name or model2.__class__.__name__
        if self.model1_info["Name"] == self.model2_info["Name"]:
            warn(f"Both models named '{self.model1_info['Name']}'. Consider unique names.")

        # Layers to hook
        self.model1_layers = model1_layers
        self.model2_layers = model2_layers
        self.model1_info["Layers"] = []
        self.model2_info["Layers"] = []

        # Feature storage
        self.model1_features: Dict[str, torch.Tensor] = {}
        self.model2_features: Dict[str, torch.Tensor] = {}

        # Hook handle storage
        self._hook_handles = []

        # Insert hooks and move to device
        self._insert_hooks()
        self.model1.to(self.device)
        self.model2.to(self.device)

    def _log_layer(
        self,
        model_key: str,
        layer_name: str,
        module: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
    ):
        if model_key == "model1":
            self.model1_features[layer_name] = out
        else:
            self.model2_features[layer_name] = out

    def _insert_hooks(self):
        # Register forward hooks for both models and save handles
        for name, layer in self.model1.named_modules():
            if self.model1_layers is None or name in self.model1_layers:
                self.model1_info["Layers"].append(name)
                handle = layer.register_forward_hook(partial(self._log_layer, "model1", name))
                self._hook_handles.append(handle)
        for name, layer in self.model2.named_modules():
            if self.model2_layers is None or name in self.model2_layers:
                self.model2_info["Layers"].append(name)
                handle = layer.register_forward_hook(partial(self._log_layer, "model2", name))
                self._hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks and clear handles."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self):
        # Ensure hooks are removed on deletion
        self.remove_hooks()

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        Kc = self._center_gram(K)
        Lc = self._center_gram(L)
        hsic_val = (Kc * Lc).sum()  # trace of dot product
        n = K.shape[0]
        return hsic_val / ((n - 1) ** 2)

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """
        Gram matrix K를 평균 0이 되도록 centering.
        H = I - 1/n * 11^T를 사용한 double centering 수행.
        """
        n = K.size(0)
        unit = torch.ones(n, n, device=K.device) / n
        H = torch.eye(n, device=K.device) - unit
        return H @ K @ H  # centered Gram matrix

    @torch.no_grad()
    def compare(
        self,
        dataloader1: DataLoader,
        dataloader2: DataLoader = None,
        infer_high_freq: bool = False,
        infer_low_freq: bool = False,
        infer_random: bool = False,
    ) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        infer_high_freq: (bool) Whether to infer high-frequency tokens.
        infer_low_freq: (bool) Whether to infer low-frequency tokens.
        infer_random: (bool) Whether to infer random tokens.
        """

        assert not (
            infer_high_freq and infer_low_freq
        ), "Either infer_high_freq or infer_low_freq must be True"

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info["Dataset"] = dataloader1.dataset.__repr__().split("\n")[0]
        self.model2_info["Dataset"] = dataloader2.dataset.__repr__().split("\n")[0]

        N = (
            len(self.model1_layers)
            if self.model1_layers is not None
            else len(list(self.model1.modules()))
        )
        M = (
            len(self.model2_layers)
            if self.model2_layers is not None
            else len(list(self.model2.modules()))
        )

        self.hsic_matrix = torch.zeros(N, M, 3, device=self.device)

        num_batches = min(len(dataloader1), len(dataloader1))

        for x1, x2 in tqdm(
            zip(dataloader1, dataloader2),
            desc="| Comparing features |",
            total=num_batches,
        ):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1.forward_features(x1.to(self.device))
            _ = self.model2.forward_features(
                x2.to(self.device),
                infer_high_freq=infer_high_freq,
                infer_low_freq=infer_low_freq,
                infer_random=infer_random,
                mask_ratio=0.25,
            )

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1[:, :1].flatten(1)
                K = X @ X.t()
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2[:, :1].flatten(1)
                    L = Y @ Y.t()
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                    if feat1.shape == feat2.shape:
                        assert not torch.all(
                            torch.eq(feat1, feat2)
                        ), f"Features are identical for {name1} and {name2}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (
            self.hsic_matrix[:, :, 0].clamp(min=1e-12).sqrt()
            * self.hsic_matrix[:, :, 2].clamp(min=1e-12).sqrt()
        )
        self.hsic_matrix = self.hsic_matrix.to("cpu")

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
        self.remove_hooks()
        self.model1_features.clear()
        self.model2_features.clear()

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info["Name"],
            "model2_name": self.model2_info["Name"],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info["Layers"],
            "model2_layers": self.model2_info["Layers"],
            "dataset1_name": self.model1_info["Dataset"],
            "dataset2_name": self.model2_info["Dataset"],
        }

    def plot_cka_matrix(self, save_path: str = None, title: str = None):
        """Plot the CKA matrix as a heatmap.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            title (str, optional): Custom title for the plot. Defaults to None.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18, pad=15)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def plot_linear_cka(self, save_path: str = None, title: str = None):
        """Plot the diagonal CKA values as a line plot.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            title (str, optional): Custom title for the plot. Defaults to None.
        """
        # Extract diagonal elements from the HSIC matrix (standard diagonal)
        n = self.hsic_matrix.shape[0]
        diagonal_cka = torch.diag(self.hsic_matrix)

        fig, ax = plt.subplots()
        ax.plot(diagonal_cka, marker="o", linestyle="-", color="b")
        ax.set_xlabel(f"Layers {self.model1_info['Name']}", fontsize=15)
        ax.set_ylabel("CKA", fontsize=15)
        # Set y-axis range from 0 to 1
        ax.set_ylim(0, 1.02)
        # Set x-axis ticks to show block numbers
        num_layers = len(diagonal_cka)
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels([f"block_{i}" for i in range(num_layers)], rotation=45)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18, pad=15)
        else:
            ax.set_title(
                f"{self.model1_info['Name']} vs {self.model2_info['Name']} (Diagonal CKA)",
                fontsize=18,
                pad=15,
            )

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()


def get_freqmae(weights_path):
    """Create and return a FreqMAE encoder backbone.

    Args:
        weights_path (str): Path to the model weights checkpoint.

    Returns:
        FreqMAEVisionTransformer: The initialized FreqMAE backbone model.
    """
    encoder = SSLEncoder(
        model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m",
        algorithm="freqmae",
        pretrained=True,
        ckpt_path=weights_path,
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
        visible_ratio=0.2506265193179896,
        mask_ratio=0.2506265193179896,
    )
    return encoder.backbone


dataframes = {
    "RFMiD": Dataframe(
        root="your_path/RFMiD",
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
    "RFMiD2": Dataframe(
        root="your_path/RFMiD2",
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
    "Drishti-GS1": Dataframe(
        root="your_path/Drishti-GS1",
        image_folder_name="images",
        label_file_name="labels-duplicate_rm",
        suffix="base_proc-224x224",
    ).dataframe,
    "APTOS": Dataframe(
        root="your_path/APTOS",
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

dataframe = pd.concat(list(dataframes.values()))


class ImageDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["filepath"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if image.max() > 1:
            image = image / 255.0

        return image


dataset = ImageDataset(dataframe)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

layer_names = ["model.blocks.11"]
ckpt_paths = [f"{WEIGHTS_DIR}/encoder_decoder-epoch_{(i+1)*10-1:07d}.ckpt" for i in range(10)]

all_results = defaultdict(list)
for i, ckpt_path in enumerate(ckpt_paths):
    model = get_freqmae(ckpt_path)
    cka_model = CKA_FreqMAE(
        deepcopy(model),
        deepcopy(model),
        model1_name="Full Token",
        model2_name="High-freq Masked Token",
        model1_layers=layer_names,
        model2_layers=layer_names,
        device="cuda",
    )
    cka_model.compare(dataloader, infer_high_freq=True)
    linear_cka = torch.diag(cka_model.hsic_matrix)
    all_results["high-freq."].append(linear_cka[0].item())
    del cka_model
    torch.cuda.empty_cache()

    cka_model = CKA_FreqMAE(
        deepcopy(model),
        deepcopy(model),
        model1_name="Full Token",
        model2_name="High-freq Masked Token",
        model1_layers=layer_names,
        model2_layers=layer_names,
        device="cuda",
    )
    cka_model.compare(dataloader, infer_low_freq=True)
    linear_cka = torch.diag(cka_model.hsic_matrix)
    all_results["low-freq."].append(linear_cka[0].item())
    del cka_model
    torch.cuda.empty_cache()

    cka_model = CKA_FreqMAE(
        deepcopy(model),
        deepcopy(model),
        model1_name="Full Token",
        model2_name="High-freq Masked Token",
        model1_layers=layer_names,
        model2_layers=layer_names,
        device="cuda",
    )
    cka_model.compare(dataloader, infer_random=True)
    linear_cka = torch.diag(cka_model.hsic_matrix)
    all_results["random"].append(linear_cka[0].item())
    del cka_model
    torch.cuda.empty_cache()

# Create final dataframe with proper structure
epochs = [f"epoch {10*(i+1)}" for i in range(len(ckpt_paths))]
res_df = pd.DataFrame(all_results, index=epochs).T
res_df.index.name = "Infer"
res_df = res_df.reset_index()

save_dir = Path(
    "your_path/visualization/cka/save"
)
save_dir.mkdir(exist_ok=True, parents=True)
model_name = Path(WEIGHTS_DIR).parent.name
res_df.to_csv(save_dir / f"cka_analysis_along_epoch_{model_name}.csv", index=False)
