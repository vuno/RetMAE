# Backbone Models

Backbone implementations and external-baseline wrappers used during RetMAE pre-training
(global-context distillation, MI estimation) and during downstream linear probing /
fine-tuning.

## Layout

```
backbone/
├── __init__.py        # Loader functions and the `MODEL_REGISTRY` dispatcher
├── base.py            # `BackboneBase` abstract class + `TimmViTBackbone`
├── mae.py             # RetMAE encoder (frequency-aware MAE, this paper)
├── mi_estimator.py    # MINE / InfoNCE estimators used by the MI loss (this paper)
├── retclip.py         # RET-CLIP encoder (used as global-context teacher)
├── retfound.py        # RETFound  (Zhou et al., 2023)
├── falir.py           # FLAIR + KeepFIT
├── urfound.py         # UrFound
├── unimedclip.py      # UniMed-CLIP
├── vilref.py          # ViLReF
└── README.md
```

## Usage

```python
from src.tasks.pretrain.backbone import (
    get_retmae, get_retclip, get_retfound, get_flair, get_urfound,
    get_model, list_available_models,
)

retmae   = get_retmae(model_name="vit_base_patch16_224")
retclip  = get_retclip()
retfound = get_retfound()

# Or via the registry
model = get_model("retmae", model_name="vit_base_patch16_224")
print(list_available_models())
```

## Components introduced by this paper

| File | Class | Role |
|---|---|---|
| `mae.py` | `MAEVisionTransformer`, `MAEDecoder` | Frequency-aware masking + MAE encoder/decoder |
| `mi_estimator.py` | `MINE`, `InfoNCE` | Mutual-information estimators for the high-frequency MI loss |

## External baselines

We ship thin wrappers around publicly released checkpoints purely to enable apples-to-apples
comparison on the downstream tasks reported in the paper. Each baseline retains the
license of its upstream release; users must obtain the original weights from the
respective authors.

| Wrapper | Upstream paper | Project |
|---|---|---|
| `retfound.py` | Zhou et al., *Nature* 2023 | https://github.com/rmaphoh/RETFound_MAE |
| `retclip.py`  | Du et al., 2024 (RET-CLIP) | https://github.com/Zoxin-DU/RET-CLIP |
| `falir.py`    | Silva-Rodríguez et al. 2024 (FLAIR) / Wu et al. 2024 (KeepFIT) | https://github.com/jusiro/FLAIR |
| `urfound.py`  | Yu et al. 2024 (UrFound) | https://github.com/yukkai/UrFound |
| `unimedclip.py` | Khattak et al. 2024 (UniMed-CLIP) | https://github.com/mbzuai-oryx/UniMed-CLIP |
| `vilref.py`   | Wu et al. 2024 (ViLReF) | https://github.com/T6Yang/ViLReF |

Checkpoint paths are resolved from the `CKPT_ROOT` environment variable; see
the *Baselines* section of the top-level `README.md` for the expected directory
layout.

## Adding a new backbone

1. Create `<name>.py` exposing a class that subclasses `BackboneBase`.
2. Register it in `__init__.py` (`MODEL_REGISTRY[<name>] = ...` and a `get_<name>()` loader).
3. Add a corresponding `configs/model/finetune/<name>.yaml` if it should be available
   for downstream evaluation.
