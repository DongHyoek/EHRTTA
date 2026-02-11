
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --use_time --use_norm_ema --align_return_weights --use_dora --cuda \
    --n_epochs 1 --scheduler --early_stop --patience 5 \
    --ckpt_dir ./results/checkpoint/fixed_crossattn \
    --metrics_dir ./results/metrics/fixed_crossattn