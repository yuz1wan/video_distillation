GPU=$1
DATA=$2
LR=$3
IPC=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_baseline.py \
--method MTT \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 3 \
--epoch_eval_train 500 \
--init real \
--syn_steps 10 \
--expert_epochs 1 \
--max_start_epoch 10 \
--lr_img=${LR} \
--lr_teacher 0.01 \
--buffer_path ./buffer/${DATA} \
--model=ConvNet3D \
--Iteration 8000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 400 \
--train_lr ;


