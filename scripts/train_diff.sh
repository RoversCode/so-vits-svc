export CUDA_VISIBLE_DEVICES=0
exp_name="sunyanzi"
torchrun \
    --nproc_per_node=1 \
    --master_port=6778 \
    --master_addr='localhost' \
    train_diff.py \
    --model_name $exp_name