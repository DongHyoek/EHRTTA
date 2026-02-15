
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --use_time --use_norm_ema --align_return_weights --use_dora --cuda \
    --scheduler --early_stop \
    --ckpt_dir ./results/checkpoint/fixed_crossattn \
    --metrics_dir ./results/metrics/fixed_crossattn \
    --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
    --n_epochs 1 --lr 0.0005 --patience 5