#!/usr/bin/env bash
# Linear-probing with a *partial* set of input patches. Used in the paper to
# study what frequency band the encoder relies on.
#
# Set MODE to one of:
#   high_freq  - keep only high-frequency patches (default)
#   low_freq   - keep only low-frequency patches
#   random25   - random masking, mask ratio 0.25
#   random75   - random masking, mask ratio 0.75
#
# Usage:
#   CKPT_PATH=/path/to/encoder.pth MODE=high_freq bash scripts/linprobe_partial.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_DEVICES=${NUM_DEVICES:-4}
MODE=${MODE:-high_freq}

if [[ -z "${CKPT_PATH:-}" ]]; then
    echo "ERROR: CKPT_PATH must point to a pre-trained RetMAE encoder." >&2
    exit 2
fi

case "$MODE" in
    high_freq)  EXTRA=(model.encoder.infer_high_freq=True) ;;
    low_freq)   EXTRA=(model.encoder.infer_low_freq=True) ;;
    random25)   EXTRA=(model.encoder.infer_random=True model.encoder.mask_ratio=0.25) ;;
    random75)   EXTRA=(model.encoder.infer_random=True model.encoder.mask_ratio=0.75) ;;
    *)          echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac

python src/finetune.py \
    experiment=finetune/retmae \
    model.encoder.ckpt_path=${CKPT_PATH} \
    trainer.max_epochs=50 \
    callbacks.early_stopping.patience=20 \
    model.transfer_type=linear_probing \
    model.learning_rate=1e-3 \
    model.warmup_epochs=10 \
    trainer.devices=${NUM_DEVICES} \
    "${EXTRA[@]}" \
    "$@"
