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
task="transfer_learning/retmae"
ckpt_paths=(
    "your_path/save/fundus_ssl/logs/ssl-retmae-42/runs/mae_0.95/checkpoints/encoder_epoch_0000099.pth"
    )
NUM_DEVICES=4
# for ckpt_path in "${ckpt_paths[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python src/transfer_learning.py \
#     experiment="${task}" \
#     model.encoder.ckpt_path="${ckpt_path}" \
#     trainer.max_epochs=50 \
#     callbacks.early_stopping.patience=20 \
#     model.transfer_type="linear_probing" \
#     model.learning_rate=1e-3 \
#     model.warmup_epochs=10 \
#     model.encoder.infer_high_freq=True \
#     trainer.devices="${NUM_DEVICES}"
# done
# for ckpt_path in "${ckpt_paths[@]}"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python src/transfer_learning.py \
#     experiment="${task}" \
#     model.encoder.ckpt_path="${ckpt_path}" \
#     trainer.max_epochs=50 \
#     callbacks.early_stopping.patience=20 \
#     model.transfer_type="linear_probing" \
#     model.learning_rate=1e-3 \
#     model.warmup_epochs=10 \
#     model.encoder.infer_random=True \
#     model.encoder.mask_ratio=0.75 \
#     trainer.devices="${NUM_DEVICES}"
# done
# for ckpt_path in "${ckpt_paths[@]}"; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python src/transfer_learning.py \
#     experiment="${task}" \
#     model.encoder.ckpt_path="${ckpt_path}" \
#     trainer.max_epochs=50 \
#     callbacks.early_stopping.patience=20 \
#     model.transfer_type="linear_probing" \
#     model.learning_rate=1e-3 \
#     model.warmup_epochs=10 \
#     model.encoder.infer_low_freq=True \
#     trainer.devices="${NUM_DEVICES}"
# done
for ckpt_path in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/transfer_learning.py \
    experiment="${task}" \
    model.encoder.ckpt_path="${ckpt_path}" \
    trainer.max_epochs=50 \
    callbacks.early_stopping.patience=20 \
    model.transfer_type="linear_probing" \
    model.learning_rate=1e-3 \
    model.warmup_epochs=10 \
    model.encoder.infer_random=True \
    model.encoder.mask_ratio=0.25 \
    trainer.devices="${NUM_DEVICES}"
done
