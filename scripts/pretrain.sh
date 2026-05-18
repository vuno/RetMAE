#!/usr/bin/env bash
# Pre-train RetMAE on the 250k retina-image corpus.
#
# Requires DATA_ROOT and CKPT_ROOT to be set in `.env` (see `.env.example`).
# Override any setting on the command line, e.g.:
#   bash scripts/pretrain.sh trainer.devices=4 model.encoder.mask_ratio=0.5
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python src/pretrain.py \
    experiment=pretrain/retmae_retclip \
    seed=42 \
    trainer.devices=2 \
    trainer.max_epochs=100 \
    model.lambda_hmi=0.01 \
    model.lambda_distill=1 \
    model.high_freq_start_epoch=0 \
    model.encoder.mask_ratio=0.3 \
    "$@"
