#!/bin/bash
export PYTHONPATH=.


CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3
CKPT_NAME=${CKPT_PATH##*/}


ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
CONV_MODE=vicuna_v1


if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/pope/llava_pope_test.jsonl \
        --image-folder $ROOT_DIR/playground/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/$CKPT_NAME-$IMAGE_SCALE.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
else
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/pope/llava_pope_test.jsonl \
        --image-folder $ROOT_DIR/playground/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/$CKPT_NAME-$IMAGE_SCALE.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
fi


python src/eval/llava/eval_pope.py \
    --annotation-dir $ROOT_DIR/playground/pope/annotations \
    --question-file $ROOT_DIR/playground/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT_NAME-$IMAGE_SCALE.jsonl