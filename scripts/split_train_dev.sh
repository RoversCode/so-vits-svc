name=sunyanzi
python preprocess/preprocess_flist_config.py  \
 --data_dir ft_ckpts/${name} \
 --speech_encoder vec768l12 \
 --vol_aug \  # 响度嵌入
