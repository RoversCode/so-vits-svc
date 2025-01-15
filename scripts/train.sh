export CUDA_VISIBLE_DEVICES=0
exp_name="雷电将军"
target_model="sovits" # sovits  diffusion
torchrun \
    --nproc_per_node=1 \
    --master_port=6791 \
    --master_addr='localhost' \
    train.py \
    --exp_name $exp_name \
    --target_model $target_model
