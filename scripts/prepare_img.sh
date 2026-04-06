export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s unlimited

# MIMIC
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/miiv \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind \
    --fig_size 672 \
    --dpi 100 \

# eICU
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/eicu \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind \
    --fig_size 672 \
    --dpi 100 \

# HIRID
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/hirid \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind \
    --fig_size 672 \
    --dpi 100 \