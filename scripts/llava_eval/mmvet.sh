#!/bin/bash

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0,1"

CKPT_PATH=$1
MODEL_BASE=$2
CKPT_NAME=${CKPT_PATH##*/}

ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
CONV_MODE=vicuna_v1


if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/mm-vet/llava-mm-vet.jsonl \
        --image-folder $ROOT_DIR/playground/mm-vet/images \
        --answers-file ./playground/data/eval/mm-vet/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE
else
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --image-scale 448 \
        --question-file $ROOT_DIR/playground/mm-vet/llava-mm-vet.jsonl \
        --image-folder $ROOT_DIR/playground/mm-vet/images \
        --answers-file ./playground/data/eval/mm-vet/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE
fi


mkdir -p ./playground/data/eval/mm-vet/results


python src/eval/llava/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$CKPT_NAME.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$CKPT_NAME.json