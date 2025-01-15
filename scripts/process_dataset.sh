#!/bin/bash

# 设置数据集名称
name="雷电将军"

echo "Step 1: Renaming and slicing audio files..."
python preprocess/rename_and_slice.py \
    --audio_dir ckpts/$name/raw_audio \
    --sr 44100 \
    --db_thresh -40 \
    --sliced_min_length 5000 \
    --min_interval 300 \
    --audio_min_length 2000 \
    --hop_size 10 \
    --max_sil_kept 500

echo "Step 2: Generating HuBERT and F0 features..."
CUDA_VISIBLE_DEVICES=0 python preprocess/gen_hubert_f0.py \
    --device cuda \
    --data_dir ckpts/$name/audio_slice \
    --f0_predictor rmvpe \
    --num_processes 3

echo "Step 3: Splitting into train and dev sets..."
python preprocess/split_train_dev.py \
    --data_dir ckpts/$name \
    --speech_encoder vec768l12 \
    --train_type finetune \
    --vol_aug

echo "All processing steps completed!" 