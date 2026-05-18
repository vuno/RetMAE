#!/usr/bin/env bash
# Linear-probing evaluation of a frozen encoder.
#
# Usage:
#   CKPT_PATH=/path/to/encoder_epoch_0000099.pth bash scripts/linprobe.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL=${MODEL:-retmae}
NUM_DEVICES=${NUM_DEVICES:-1}
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-20}
LR=${LR:-1e-3}
WARMUP=${WARMUP:-10}

OVERRIDES=(
    experiment=finetune/${MODEL}
    trainer.max_epochs=${MAX_EPOCHS}
    callbacks.early_stopping.patience=${PATIENCE}
    model.transfer_type=linear_probing
    model.learning_rate=${LR}
    model.warmup_epochs=${WARMUP}
    trainer.devices=${NUM_DEVICES}
    trainer.accelerator=gpu
)

if [[ "$MODEL" == "retmae" ]]; then
    if [[ -z "${CKPT_PATH:-}" ]]; then
        echo "ERROR: CKPT_PATH must point to a pre-trained RetMAE encoder for MODEL=retmae." >&2
        exit 2
    fi
    OVERRIDES+=(model.encoder.ckpt_path=${CKPT_PATH})
fi

python src/finetune.py "${OVERRIDES[@]}" "$@"
