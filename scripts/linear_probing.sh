#!/bin/bash

# task=(
#     "transfer_learning/flair"
#     "transfer_learning/keepfit"
#     "transfer_learning/retclip"
#     "transfer_learning/urfound"
#     "transfer_learning/retfound"
# )

# NUM_DEVICES=2
# for task in "${task[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1 python src/transfer_learning.py \
#     experiment="${task}" \
#     trainer.max_epochs=50 \
#     callbacks.early_stopping.patience=20 \
#     model.transfer_type="linear_probing" \
#     model.learning_rate=1e-3 \
#     model.warmup_epochs=10 \
#     trainer.devices="${NUM_DEVICES}"
# done

# RetMAE
DEVICE=1
task="transfer_learning/retmae"
ckpt_paths=(
        "your_path/save/fundus_ssl/logs/ssl-retmae_retclip-42/runs/retmae_retclip-1_hmi-0.01-60_m-0.3/checkpoints/encoder_epoch_0000099.pth"
        )
for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES="${DEVICE}" \
    python src/transfer_learning.py \
    experiment="${task}" \
    model.encoder.ckpt_path="${ckpt_path}" \
    trainer.max_epochs=50 \
    callbacks.early_stopping.patience=20 \
    model.transfer_type="linear_probing" \
    model.learning_rate=1e-3 \
    model.warmup_epochs=10 \
    trainer.accelerator="gpu"
    done
