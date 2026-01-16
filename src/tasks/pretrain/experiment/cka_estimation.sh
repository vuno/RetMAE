#!/bin/bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

model_ckpts=(
    # weight path
)
model_names=(
    # "MAE"
    "FunMMA"
)
for ((i=0; i<${#model_ckpts[@]}; i++)); do
    model_ckpt="${model_ckpts[$i]}"
    python cka_estimation.py \
      --model_ckpt_99 "${model_ckpt}" \
      --batch_size 256 \
      --gpu 1 \
      --snapshot_epochs 19,49,99 \
      --model_name "${model_names[$i]}" \
      --prefix "${model_names[$i]}" \
      --save_mats_all
done
