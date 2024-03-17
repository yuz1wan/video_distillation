GPU=$1
DATA=$2
METHOD=$3

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python coreset.py \
--dataset ${DATA} \
--num_eval 3 \
--epoch_eval_train 500 \
--lr_net 0.01 \
--ipc 1 \
--model ConvNet3D \
--method ${METHOD} \