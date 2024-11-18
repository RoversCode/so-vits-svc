export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
exp_name="test"
torchrun \
    --nproc_per_node=7 \
    --master_port=6789 \
    --master_addr='localhost' \
    train.py \
    --exp_name $exp_name