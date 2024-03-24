GPU=$1
DATA=$2

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python buffer.py \
--buffer_path ./buffer/${DATA} \
--lr_teacher 0.01 \
--dataset ${DATA} \
--num_experts 30 \
--model ConvNet3D \

