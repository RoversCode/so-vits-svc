name=sunyanzi
python preprocess/split_train_dev.py  \
 --data_dir ckpts/${name} \
 --speech_encoder vec768l12 \
 --train_type finetune \
 --vol_aug # 响度嵌入
