config=$1
declare -u expname
expname=`basename ${config} .yaml`

CUDA_VISIBLE_DEVICES=0 python main.py \
--config $1 \
--output results/${expname}
