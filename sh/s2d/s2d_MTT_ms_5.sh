GPU=$1
DATA=$2

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_s2d_ms.py \
--method MTT \
--dataset ${DATA} \
--num_eval 3 \
--spc 10 \
--dpc 10 \
--vpc 5 \
--epoch_eval_train 500 \
--syn_steps 5 \
--expert_epochs 1 \
--max_start_epoch 10 \
--lr_dynamic=1e4 \
--lr_hal=1e-3 \
--lr_teacher 0.01 \
--buffer_path ./buffer/${DATA} \
--Iteration  10000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 200 \
--no_train_static \
--path_static path_to_static_memory \
--save_path ./result/ \
--batch_train 128 \
--batch_syn 128 \
--startIt 200 \


