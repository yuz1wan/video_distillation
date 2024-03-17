GPU=$1
DATA=$2
L_D=$3
L_H=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_s2d_ms.py \
--method MTT \
--dataset ${DATA} \
--num_eval 3 \
--spc 2 \
--dpc 2 \
--vpc 1 \
--epoch_eval_train 500 \
--syn_steps 10 \
--expert_epochs 1 \
--max_start_epoch 10 \
--lr_dynamic=${LR_D} \
--lr_hal=${LR_H} \
--lr_teacher 0.01 \
--buffer_path ./buffer/${DATA} \
--model=ConvNet3D \
--Iteration  10000 \
--model ConvNet3D \
--eval_it 400 \
--no_train_static \
--path_static path_to_static_memory \
--save_path ./result/ \
--startIt 400 \
--batch_train 256 \
--train_lr ;

