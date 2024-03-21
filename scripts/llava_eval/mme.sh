#!/bin/bash
export PYTHONPATH=.


CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3

CKPT_NAME=${CKPT_PATH##*/}
ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
MODEL_NAME=vicuna-7b-v1.5
CONV_MODE=vicuna_v1


if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/MME/llava_mme.jsonl \
        --image-folder $ROOT_DIR/playground/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --image-scale $IMAGE_SCALE
else
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/MME/llava_mme.jsonl \
        --image-folder $ROOT_DIR/playground/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --image-scale $IMAGE_SCALE
fi

cd ./playground/data/eval/MME

python /share/LLM_project/vlm-pretrain/unicode/src/eval/llava/convert_answer_to_mme.py --experiment $CKPT_NAME

cd eval_tool

python calculation.py --results_dir answers/$CKPT_NAME