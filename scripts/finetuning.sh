#!/bin/bash

task=(
    "transfer_learning/flair"
    "transfer_learning/keepfit"
    "transfer_learning/retclip"
    "transfer_learning/urfound"
    "transfer_learning/retfound"
    )

NUM_DEVICES=2
for task in "${task[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 \
    python src/transfer_learning.py \
    experiment="${task}" \
    trainer.max_epochs=50 \
    callbacks.early_stopping.patience=20 \
    model.learning_rate=1e-3 \
    model.warmup_epochs=10 \
    trainer.devices="${NUM_DEVICES}"
done

# # RetMAE
# task="transfer_learning/retmae"
# ckpt_paths=(
#     "your_path/save/fundus_ssl/logs/ssl-retmae_16-42/runs/gmi_0.1_retclip_hmi_0.1_sub_0.01/checkpoints/encoder_epoch_0000099.pth"
# )
# for ckpt_path in "${ckpt_paths[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1 \
#     python src/transfer_learning.py \
#     experiment="${task}" \
#     model.encoder.ckpt_path="${ckpt_path}" \
#     trainer.max_epochs=50 \
#     callbacks.early_stopping.patience=20 \
#     model.transfer_type="finetuning" \
#     model.learning_rate=1e-3 \
#     model.warmup_epochs=10 \
#     trainer.devices="${NUM_DEVICES}"
# done
