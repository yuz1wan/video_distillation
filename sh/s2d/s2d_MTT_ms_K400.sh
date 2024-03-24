GPU=$1
LR_D=$2
LR_H=$3

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_s2d_ms.py \
--method MTT \
--dataset Kinetics400 \
--num_eval 3 \
--spc 2 \
--dpc 2 \
--vpc 1 \
--epoch_eval_train 500 \
--syn_steps 10 --expert_epochs 1 \
--max_start_epoch 10 \
--lr_dynamic=${LR_D} \
--lr_hal=${LR_H} \
--lr_teacher 0.01 \
--buffer_path path_to_buffer \
--model=ConvNet3D \
--Iteration 10000 \
--model ConvNet3D \
--eval_it 1000 \
--no_train_static \
--path_static path_to_static_memory \
--batch_train 256 \
--batch_syn 256 \
--save_path ./result/ \
--eval_mode top5 \  # for k400 & ssv2 
--frames 8 \  # for k400 & ssv2 

