nohup_file_name='nohup_soft_2022_5_13_baseline_3.log'

l2=0.1             # 0.1 / 0.05
warmup_ratio=0.1    # 0.1 / 0.04 
CL_mode='soft'
max_epoch=300
lr=1e-4
batch_size=20

CL_epoch_num=40
soft_CL_matrix_decay=1

seed=400
export CUDA_VISIBLE_DEVICES=0

# CL experiment
nohup python3 -u scripts/text2sql.py --seed $seed --l2 $l2 --lr $lr --CL_epoch_num $CL_epoch_num --batch_size $batch_size \
                --soft_CL_matrix_decay $soft_CL_matrix_decay \
                --warmup_ratio $warmup_ratio --CL_mode $CL_mode \
                --max_epoch $max_epoch > nohup/$nohup_file_name 2>&1 &

# baseline experiment
# nohup python3 -u scripts/text2sql.py --seed $seed --l2 $l2 --lr $lr --CL_epoch_num $CL_epoch_num --batch_size $batch_size \
#                 --soft_CL_matrix_decay $soft_CL_matrix_decay \
#                 --warmup_ratio $warmup_ratio \
#                 --max_epoch $max_epoch > nohup/$nohup_file_name 2>&1 &

# no nohup ; count params num
# model=lgesql
# plm=bert-large-uncased-whole-word-masking
# plm=bert-base-uncased
# python3 -u scripts/text2sql.py --seed $seed --l2 $l2 --lr $lr --CL_epoch_num $CL_epoch_num --batch_size $batch_size \
#                 --soft_CL_matrix_decay $soft_CL_matrix_decay \
#                 --warmup_ratio $warmup_ratio \
#                 --max_epoch $max_epoch --model $model --plm $plm