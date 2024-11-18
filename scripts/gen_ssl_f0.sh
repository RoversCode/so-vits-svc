# use_diff，Whether to use the diffusion model

name=sunyanzi
CUDA_VISIBLE_DEVICES=0 python preprocess/gen_hubert_f0.py  \
 --device cuda \
 --data_dir ckpts/$name/audio_slice \
 --use_diff  \
 --f0_predictor rmvpe \
 --num_processes 3 \
