#!/usr/bin/env bash

DATA_DIR=../dataset/ECommerce-IC
DICT_FILE=../utils/cn_tokenizer/dict.txt
SAVA_DIR=../checkpoints/ECommerce-IC
USER_DIR=../user_module
RESULT=../results/ECommerce-IC

CUDA_VISIBLE_DEVICES=0 python ../generate.py ${DATA_DIR} \
--user-dir ${USER_DIR} \
--dict-file ${DICT_FILE} \
--task caption \
--path ${SAVA_DIR}/checkpoint_last.pt \
--beam 5 \
--batch-size 32 \
--max-len-b 30 \
--results-path ${RESULT} \
--seed 7 \
--num-workers 4