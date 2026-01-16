import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 경로 설정
sys.path.append("your_path")
from src.common.data.base import Dataframe
from src.tasks.pretrain.utils import butterworth_high_pass_filter, soft_edge_mask

# 파라미터
BEST_PARAMS = {
    "butterworth_cutoff": 14.0470391901718,
    "butterworth_order": 2,
    "soft_mask_th": 0.0868832016680601,
    "soft_mask_sigma": 10.133165328972893,
    "boundary_cutoff": 0.6701278573157876,
    "blur_sigma": 0.5185059445302515,
    "blur_radius": 1,
}


# ✅ 메인 함수 (히스토그램 기반 quantile 계산)
def compute_global_quantiles(dataloader, output_dir: Path, flush_interval: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 히스토그램 설정: [0, 1] 구간, 1000 bins
    bin_edges = np.linspace(0, 1, 1001)
    hist = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    total_pixels = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        g = batch[:, 1:2].to(device)  # Green 채널

        with torch.no_grad():
            high_freq, _, _ = butterworth_high_pass_filter(
                g,
                fov_mask=soft_edge_mask(
                    g,
                    threshold_frac=BEST_PARAMS["soft_mask_th"],
                    sigma=BEST_PARAMS["soft_mask_sigma"],
                ),
                cutoff=BEST_PARAMS["butterworth_cutoff"],
                order=BEST_PARAMS["butterworth_order"],
                boundary_cut=BEST_PARAMS["boundary_cutoff"],
                blur_sigma=BEST_PARAMS["blur_sigma"],
                blur_radius=BEST_PARAMS["blur_radius"],
                normalize=False,
            )
            vals = high_freq.abs().squeeze(1).flatten(1).cpu().numpy()
            batch_hist, _ = np.histogram(vals, bins=bin_edges)
            hist += batch_hist
            total_pixels += vals.size

        del g, high_freq
        torch.cuda.empty_cache()

    # ✅ 누적 히스토그램 → CDF → quantile 계산
    cdf = np.cumsum(hist) / total_pixels
    q25 = bin_edges[np.searchsorted(cdf, 0.25)]
    median = bin_edges[np.searchsorted(cdf, 0.5)]
    q75 = bin_edges[np.searchsorted(cdf, 0.75)]

    # 결과 저장 (CSV)
    np.savetxt(
        output_dir / "imagenet_global_highfreq_quantiles.csv",
        np.array([["q25", q25], ["median", median], ["q75", q75]]),
        delimiter=",",
        fmt="%s",
    )

    print(f"Saved global quantiles to {output_dir / 'imagenet_global_highfreq_quantiles.csv'}")


# ✅ ImageNet Dataset 정의
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


# ✅ DataLoader 구성
imagenet_loader = DataLoader(
    ImageNetDataset("your_path/IMGNET/1K/train"),
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

# ✅ 실행
if __name__ == "__main__":
    output_dir = Path(__file__).parent
    compute_global_quantiles(imagenet_loader, output_dir)
