GPU=$1
LR=$2
LR_H=$3

CUDA_VISIBLE_DEVICES=${GPU} python distill_s2d.py --lr_d ${LR} --lr_h ${LR_H} --model ConvNet3D --dataset miniUCF101 \
--epoch_eval_train 1000 --eval_it 200 --learn_label --lr_net 0.0001 --num_eval 3 --startIt 200 \
--path_static path_to_static_memory \
--spc 5 --dpc 5 --num_prototypes_per_class 5 \