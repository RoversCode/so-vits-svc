#!/bin/bash

# 设置实验名称和设备ID
exp_name="八重神子"
export CUDA_VISIBLE_DEVICES=0

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