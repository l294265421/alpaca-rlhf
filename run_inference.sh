#!/usr/bin/env bash
# nohup sh run.sh chat_sentiment_analysis/llama/finetune.py > autodl.log 2>&1 &
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

 gpu_card=$1
 shift
 export CUDA_VISIBLE_DEVICES=${gpu_card}

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

python $@


end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
