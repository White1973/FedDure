#!/bin/sh
GPUS_PER_NODE=16

PARTITION=$1
GPUS=$2
config=$3

declare -u expname
expname=`basename ${config} .yaml`

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

currenttime=`date "+%Y%m%d%H%M%S"`
hostname='SH-IDC1-10-198-9-[8]'
mkdir -p  results/${expname}/train_log
g=$(($2<16?$2:16))
set -x

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${expname} \
    --gres=gpu:$g \
    --ntasks=${GPUS} \
    --ntasks-per-node=$g \
    --cpus-per-task=${CPUS_PER_TASK} \
    -x $hostname \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u -W ignore main.py \
    --config $3 \
    --output results/${expname}
    ${PY_ARGS} \
    2>&1 | tee results/${expname}/train_log/train_${currenttime}.log


#sh ~/dotfiles/tools/install.sh --zsh-offline --tmux --gitconfig
    #-x $hostname \
    #--quotatype=auto \
    #-x $hostname \
