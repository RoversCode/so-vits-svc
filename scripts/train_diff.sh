export CUDA_VISIBLE_DEVICES=0
exp_name="gensin"
target_model="diffusion"
torchrun \
    --nproc_per_node=1 \
    --master_port=6976 \
    --master_addr='localhost' \
    train_diff.py \
    --exp_name $exp_name \
    --target_model $target_model