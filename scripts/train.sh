
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s unlimited

python main.py \
    --use_time  --use_norm_ema --align_return_weights --use_dora --cuda \
    --scheduler --early_stop \
    --ckpt_dir ./results/checkpoint/align_with_vocab_timellmstyle_w_focal \
    --metrics_dir ./results/metrics/align_with_vocab_timellmstyle_w_focal \
    --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
    --n_epochs 50 --lr 0.0001 --patience 5 \
    --batch_size 8 --h_pool last \
    --model_id meta-llama/Llama-3.2-1B \
    --use_ts_trunc --max_length 60 --te_cls_init raw_tok

## Dora 사용 X (LoRA로만 활용)
# python -X faulthandler main.py \
#     --use_time  --use_norm_ema --align_return_weights --cuda \
#     --scheduler --early_stop \
#     --ckpt_dir ./results/checkpoint/change_init_text_enc_wo_dora \
#     --metrics_dir ./results/metrics/change_init_text_enc_wo_dora \
#     --te_id_mix --te_n_vars 45 --te_n_fields 11 --te_n_texts 414 \
#     --n_epochs 50 --lr 0.001 --patience 5 \
#     --batch_size 8 --h_pool last \
#     --model_id meta-llama/Llama-3.2-1B \
#     --use_ts_trunc --max_length 60 --te_cls_init raw_tok \
