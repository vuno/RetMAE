#!/bin/bash
max_epoch=100
global_ratios=(1)
hf_ratios=(0.01)
high_freq_start_epochs=(0)
mask_ratios=(0.3)
experiment=pretrain/retmae_retclip

for i in "${!global_ratios[@]}"; do
    for high_freq_start_epoch in "${high_freq_start_epochs[@]}"; do
        for mask_ratio in "${mask_ratios[@]}"; do
            echo "Training retmae with max_epoch=${max_epoch} and hf_ratio=${hf_ratios[$i]} and global_ratio=${global_ratios[$i]} and mask_ratio=${mask_ratio} and high_freq_start_epoch=${high_freq_start_epoch}"
            CUDA_VISIBLE_DEVICES=0,1 \
                python src/train.py \
                trainer.devices=2 \
                experiment="${experiment}" \
                seed=42 \
                trainer.max_epochs="${max_epoch}" \
                model.lambda_hmi="${hf_ratios[$i]}" \
                model.lambda_distill="${global_ratios[$i]}" \
                model.high_freq_start_epoch="${high_freq_start_epoch}" \
                model.encoder.mask_ratio="${mask_ratio}"
        done
    done
done
