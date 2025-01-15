# use_diff，Whether to use the diffusion model
# rmvpe # crepe # pm # harvest # dio

name=可莉
CUDA_VISIBLE_DEVICES=0 python preprocess/gen_hubert_f0.py  \
 --device cuda \
 --data_dir ckpts/$name/audio_slice \
 --f0_predictor rmvpe \
 --num_processes 3 \
