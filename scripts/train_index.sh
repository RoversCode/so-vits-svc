export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=32  # 或者更小的数字
export MKL_NUM_THREADS=32
export OMP_NUM_THREADS=32
exp_name="芙宁娜"
target_model="sovits"
python train_index.py --exp_name $exp_name --target_model $target_model