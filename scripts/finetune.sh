#!/usr/bin/env bash
# End-to-end fine-tuning of a pre-trained encoder on the downstream tasks
# (DR / AMD / Glaucoma / RVO).
#
# Usage:
#   CKPT_PATH=/path/to/encoder_epoch_0000099.pth bash scripts/finetune.sh
#
# Override `MODEL` to fine-tune one of the baseline encoders we ship in this
# repository, e.g. `MODEL=retfound bash scripts/finetune.sh`. Available:
# retmae, retfound, retclip, flair, keepfit, urfound, unimedclip, vilref.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
MODEL=${MODEL:-retmae}
NUM_DEVICES=${NUM_DEVICES:-2}
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-20}
LR=${LR:-1e-3}
WARMUP=${WARMUP:-10}

OVERRIDES=(
    experiment=finetune/${MODEL}
    trainer.max_epochs=${MAX_EPOCHS}
    callbacks.early_stopping.patience=${PATIENCE}
    model.transfer_type=finetuning
    model.learning_rate=${LR}
    model.warmup_epochs=${WARMUP}
    trainer.devices=${NUM_DEVICES}
)

if [[ "$MODEL" == "retmae" ]]; then
    if [[ -z "${CKPT_PATH:-}" ]]; then
        echo "ERROR: CKPT_PATH must point to a pre-trained RetMAE encoder for MODEL=retmae." >&2
        exit 2
    fi
    OVERRIDES+=(model.encoder.ckpt_path=${CKPT_PATH})
fi

python src/finetune.py "${OVERRIDES[@]}" "$@"
