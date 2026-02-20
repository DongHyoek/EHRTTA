
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --use_time --use_norm_ema --align_return_weights --use_dora --cuda \
    --scheduler --early_stop \
    --ckpt_dir ./results/checkpoint/1b_model \
    --metrics_dir ./results/metrics/1b_model \
    --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
    --n_epochs 50 --lr 0.0005 --patience 5 \
    --batch_size 16 --h_pool last \
    --model_id meta-llama/Llama-3.2-1B