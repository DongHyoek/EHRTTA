# MIMIC
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/miiv \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind

# eICU
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/eicu \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind

# HIRID
python -m preprocess.transform_ts2img \
    --data_path ~/EHRTTA/data/hirid \
    --freq_min 5 \
    --min_clip -3 \
    --max_clip 3 \
    --scatter_ind