export CUDA_VISIBLE_DEVICES=0
exp_name="gensin"
target_model="sovits"
python train_index.py --exp_name $exp_name --target_model $target_model