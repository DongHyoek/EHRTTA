
export CUDA_LAUNCH_BLOCKING=1

python main.py \
    --use_time --use_norm_ema --align_return_weights --use_align_gate --use_dora --cuda \
    --n_epochs 1