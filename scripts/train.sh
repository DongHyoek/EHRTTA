
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s unlimited

# # LLM쪽은 아예 Freeze하고 Time series embedding을 바꿔보자
# python main.py \
#     --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
#     --scheduler --early_stop \
#     --ckpt_dir ./results/checkpoint/modify_concatenation_variable_embeddings \
#     --metrics_dir ./results/metrics/modify_concatenation_variable_embeddings \
#     --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
#     --n_epochs 50 --lr 0.0001 --patience 5 \
#     --batch_size 8 --h_pool last \
#     --model_id meta-llama/Llama-3.2-1B \
#     --use_ts_trunc --max_length 60 --te_cls_init raw_tok --ts_dim 32

# # Adapter의 적용되는 영역을 늘리기
# python main.py \
#     --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
#     --scheduler --early_stop \
#     --ckpt_dir ./results/checkpoint/expand_adapter_position_wo_aggregator \
#     --metrics_dir ./results/metrics/expand_adapter_position_wo_aggregator \
#     --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
#     --n_epochs 50 --lr 0.0001 --patience 5 \
#     --batch_size 8 --h_pool last \
#     --model_id meta-llama/Llama-3.2-1B \
#     --use_ts_trunc --max_length 60 --te_cls_init raw_tok --ts_dim 32 \


# Adapter의 적용되는 영역을 늘리고 다시 independent한 구조로 학습한 뒤 aggregator 추가하기
python main.py \
    --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
    --scheduler --early_stop \
    --ckpt_dir ./results/checkpoint/add_optimizing_params \
    --metrics_dir ./results/metrics/add_optimizing_params \
    --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
    --n_epochs 50 --lr 0.0001 --patience 5 \
    --batch_size 8 --h_pool last \
    --model_id meta-llama/Llama-3.2-1B \
    --use_ts_trunc --max_length 60 --te_id_mix --te_cls_init raw_tok --ts_dim 2048 \

