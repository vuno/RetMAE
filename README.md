# RetMAE

This is an implementation of 'Frequency-Balanced Retinal Representation Learning with Mutual Information Regularization'.

## Environments

### Prerequisites
- Conda (Miniconda or Anaconda)

### Installation
The project environment is managed by Conda and defined in `src/tasks/pretrain/environment.yaml`.

1. Create the conda environment:
```bash
conda env create -f src/tasks/pretrain/environment.yaml
```

2. Activate the environment:
```bash
conda activate fundus-pretrain
```

### Key Dependencies
- python=3.11
- pytorch=2.4.1 (cuda 12.1)
- torchvision=0.19.1
- lightning=2.4
- timm>=1.0.12
- hydra-core=1.3.2
- torchmetrics=1.5.2
- kornia
- einops
- openvino=2025.0.0
- aim>=3.16.2
- mlflow

## Usage

### Pre-training
To pre-train RetMAE:
```bash
bash scripts/train-retmae_retclip.sh
```

### Fine-tuning
To fine-tune the model:
```bash
bash scripts/finetuning.sh
```

## License
© VUNO Inc. All rights reserved.

This repository contains code developed at VUNO Inc. by its employees as part of their official duties.
Do not distribute, modify, or use this code outside the scope permitted by the license without explicit permission from VUNO.
