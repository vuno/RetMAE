import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("your_path")
from src.common.data.base import Dataframe
from src.tasks.pretrain.utils import butterworth_high_pass_filter, soft_edge_mask

# 공통 파라미터
BEST_PARAMS = {
    "butterworth_cutoff": 14.0470391901718,
    "butterworth_order": 2,
    "soft_mask_th": 0.0868832016680601,
    "soft_mask_sigma": 10.133165328972893,
    "boundary_cutoff": 0.6701278573157876,
    "blur_sigma": 0.5185059445302515,
    "blur_radius": 1,
}

# 공통 histogram bins 정의 (0~1, 50개 구간)
bin_edges = np.linspace(0, 1, 101, endpoint=True)[1:]
bin_edges = np.insert(bin_edges, 0, 0.0)


def compute_and_save_histogram(dataloader, save_path):
    # 1) 빈도 누적용 배열 초기화
    bin_width = bin_edges[1] - bin_edges[0]
    hist_count = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    total_pixels = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in tqdm(dataloader, desc=f"Processing {save_path.name}"):
        # green 채널만 사용
        g = batch[:, 1:2].to(device)

        # high-pass filter
        high_freq, _, _ = butterworth_high_pass_filter(
            g,
            fov_mask=soft_edge_mask(
                g, threshold_frac=BEST_PARAMS["soft_mask_th"], sigma=BEST_PARAMS["soft_mask_sigma"]
            ),
            cutoff=BEST_PARAMS["butterworth_cutoff"],
            order=BEST_PARAMS["butterworth_order"],
            boundary_cut=BEST_PARAMS["boundary_cutoff"],
            blur_sigma=BEST_PARAMS["blur_sigma"],
            blur_radius=BEST_PARAMS["blur_radius"],
            normalize=False,
        )

        # 절댓값, CPU로 옮겨 1D 배열로 변환
        vals = high_freq.squeeze(1).flatten(1).cpu().numpy()

        # 배치별 히스토그램 계산 후 누적
        batch_count, _ = np.histogram(vals, bins=bin_edges)
        hist_count += batch_count
        total_pixels += vals.size

        # vals는 더 이상 참조되지 않으므로 GC 대상이 됩니다.

    # 2) 밀도(density) 계산: density = count / (전체 픽셀 수 * bin_width)
    hist_density = hist_count.astype(float) / (total_pixels * bin_width)

    # 3) DataFrame으로 저장
    df = pd.DataFrame({"value": bin_edges[:-1], "density": hist_density})
    df.to_csv(save_path, index=False)
    print(f"Saved histogram to {save_path}")


# 1) ImageNet val histogram
class ImageNetDataset(Dataset):
    def __init__(self, datadir):
        self.files = list(Path(datadir).rglob("*.JPEG"))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        t = self.transform(img)
        return t / 255.0 if t.max() > 1 else t


imagenet_loader = DataLoader(
    ImageNetDataset("your_path/IMGNET/1K/train"),
    batch_size=512,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)
compute_and_save_histogram(imagenet_loader, Path(__file__).parent / "high_freq_imgnet.csv")

dataset_list = [
    "DiaRetDB2",
    "OIA-DDR",
    "REFUGE",
    "1000fundus",
    "DR1-2",
    "DRD",
    "drimdb",
    "DRIONS-DB",
    "FGADR",
    "FIRE",
    "FUND-OCT",
    "G1020",
    "HEI-MED",
    "HRF",
    "IOSTAR",
    "JICHI",
    "JustRAIGS",
    "LAG",
    "LES",
    "MSHF",
    "ODIR",
    "ORIGA",
    "PALM",
    "PAPILA",
    "RC-RGB-MA",
    "RetinalLesion",
    "RIDB",
    "RIGA",
    "ROC",
    "SUSTech-SYSU",
    "TREND",
    "VICAVR",
    "AGAR300",
    "ARIA",
    "AVRDB",
    "Benitez",
    "BRSET",
    "Cataract",
    "DeepDRiD",
    "DiaRetDB1",
]

dfs = {
    name: Dataframe(
        root=f"your_path/{name}",
        image_folder_name="images",
        label_file_name="labels-duplicate_rm",
        suffix="base_proc-224x224",
    ).dataframe
    for name in dataset_list
}
fundus_df = pd.concat(dfs.values(), ignore_index=True)


class FundusDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["filepath"]).convert("RGB")
        t = self.transform(img)
        return t / 255.0 if t.max() > 1 else t


fundus_loader = DataLoader(
    FundusDataset(fundus_df), batch_size=512, shuffle=False, num_workers=16, pin_memory=True
)
compute_and_save_histogram(fundus_loader, Path(__file__).parent / "high_freq_fundus.csv")
