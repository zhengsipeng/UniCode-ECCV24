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
        --question-file $ROOT_DIR/playground/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $ROOT_DIR/playground/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
else
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $ROOT_DIR/playground/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
fi

python -m src.eval.llava.eval_textvqa \
    --annotation-file $ROOT_DIR/playground/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT_NAME.jsonl
