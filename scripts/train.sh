
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s unlimited

# LLM쪽은 아예 Freeze
python main.py \
    --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
    --scheduler --early_stop \
    --ckpt_dir ./results/checkpoint/del_adapter_non_independent \
    --metrics_dir ./results/metrics/del_adapter_non_independent \
    --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
    --n_epochs 50 --lr 0.0001 --patience 5 \
    --batch_size 8 --h_pool last \
    --model_id meta-llama/Llama-3.2-1B \
    --use_ts_trunc --max_length 60 --te_cls_init raw_tok --ts_dim 32

# # Adapter의 적용되는 영역을 늘리기
# python main.py \
#     --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
#     --scheduler --early_stop \
#     --ckpt_dir ./results/checkpoint/extend_adapter_position \
#     --metrics_dir ./results/metrics/extend_adapter_position \
#     --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
#     --n_epochs 50 --lr 0.0001 --patience 5 \
#     --batch_size 8 --h_pool last \
#     --model_id meta-llama/Llama-3.2-1B \
#     --use_ts_trunc --max_length 60 --te_cls_init raw_tok --ts_dim 32 \
#     --target_module ('q_proj' 'k_proj' 'v_proj', 'o_proj')

# # align을 vocab과 함께하되 summary token 없는 mean구조
# python main.py \
#     --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
#     --scheduler --early_stop \
#     --ckpt_dir ./results/checkpoint/wo_learnable_summarize_tokens_mean_pool \
#     --metrics_dir ./results/metrics/wo_learnable_summarize_tokens_mean_pool \
#     --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
#     --n_epochs 50 --lr 0.0001 --patience 5 \
#     --batch_size 8 --h_pool mean \
#     --model_id meta-llama/Llama-3.2-1B \
#     --use_ts_trunc --max_length 60 --te_cls_init raw_tok

