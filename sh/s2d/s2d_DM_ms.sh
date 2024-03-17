GPU=$1
DATA=$2
L_D=$3
L_H=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_s2d_ms.py \
--method DM \
--dataset ${DATA} \
--num_eval 3 \
--vpc 1 \
--spc 2 \
--dpc 2 \
--epoch_eval_train 500 \
--lr_dynamic=${L_D} \
--lr_hal=${L_H} \
--model=ConvNet3D \
--batch_real 64 \
--Iteration  5000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 400 \
--no_train_static \
--path_static path_to_static_memory \
--startIt 400 \
--preload \
--save_path ./result/ \
