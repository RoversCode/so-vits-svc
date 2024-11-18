python preprocess/preprocess_flist_config.py  \
 --train_list ./filelists/train.txt \
 --val_list ./filelists/val.txt \
 --source_dir dataset_raw \
 --speech_encoder vec768l12 \
 --vol_aug \  # 响度嵌入
