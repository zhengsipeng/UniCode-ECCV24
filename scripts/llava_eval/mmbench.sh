#!/bin/bash
export PYTHONPATH=.


ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
SPLIT=mmbench_dev_20230712

CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3
CKPT_NAME=${CKPT_PATH##*/}
CONV_MODE=vicuna_v1

IMG_SCALE=0
if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_mmbench \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT_NAME-$IMAGE_SCALE.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
else
    python -m src.eval.llava.model_vqa_mmbench \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT_NAME-$IMAGE_SCALE.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
fi

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT


python -m src.eval.llava.convert_mmbench_for_submission \
    --annotation-file $ROOT_DIR/playground/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT_NAME-$IMAGE_SCALE