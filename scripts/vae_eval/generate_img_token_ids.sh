#!/bin/bash

export PYTHONPATH=.

CKPT_PATH=$1
DATASET=$2
GPU_ID=$3

CKPT_NAME=${CKPT_PATH##*/}


CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval/vae/generate_img_token_ids.py -r=$CKPT_PATH -d=$DATASET --fid --split train -b 64

