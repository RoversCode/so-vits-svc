#!/bin/bash

# 设置实验名称和设备ID
exp_name="芙宁娜"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=32  # 或者更小的数字
export MKL_NUM_THREADS=32
export OMP_NUM_THREADS=32

# 1. 数据预处理
echo "Step 1: Preprocessing data..."

# 1.1 重命名和切片
python preprocess/rename_and_slice.py  \
 --audio_dir ckpts/$exp_name/raw_audio \
 --sr 44100 \
 --db_thresh -40 \
 --sliced_min_length 5000 \
 --min_interval 300 \
 --audio_min_length 2000 \
 --hop_size 10 \
 --max_sil_kept 500

# 1.2 生成HuBERT特征和F0
python preprocess/gen_hubert_f0.py \
    --device cuda \
    --data_dir ckpts/${exp_name}/audio_slice \
    --f0_predictor rmvpe \
    --num_processes 3


# 1.3 生成配置文件
python preprocess/split_train_dev.py  \
 --data_dir ckpts/${exp_name} \
 --speech_encoder vec768l12 \
 --train_type finetune \
 --vol_aug # 响度嵌入

# 2. 训练基础模型
echo "Step 2: Training base model..."
torchrun \
    --nproc_per_node=1 \
    --master_port=6791 \
    --master_addr='localhost' \
    train.py \
    --exp_name $exp_name \
    --target_model sovits


# 3. 训练浅扩散模型
echo "Step 3: Training shallow diffusion model..."
torchrun \
    --nproc_per_node=1 \
    --master_port=6976 \
    --master_addr='localhost' \
    train_diff.py \
    --exp_name $exp_name \
    --target_model diffusion

# 4. 特征检索模型训练(可选)
echo "Step 4: Training feature retrieval model..."
python train_index.py --exp_name $exp_name --target_model sovits
echo "Training complete!"

# 打印训练完成的模型路径
echo "Trained models saved at:"
echo "Base model: logs/${exp_name}/G_*.pth"
echo "Diffusion model: logs/${exp_name}/diffusion/model_*.pt"
echo "Feature index: logs/${exp_name}/feature_and_index.pkl"

# 添加简单的错误检查
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Error occurred during training!"
    exit 1
fi 