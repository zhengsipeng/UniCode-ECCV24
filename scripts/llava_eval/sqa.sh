#!/bin/bash
export PYTHONPATH=.


CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3
CKPT_NAME=${CKPT_PATH##*/}

ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
CONV_MODE=vicuna_v1



if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_science \
        --model-path $CKPT_PATH \
        --image-folder $ROOT_DIR/playground//scienceqa/test \
        --question-file $ROOT_DIR/playground/scienceqa/llava_test_CQM-A.json \
        --answers-file ./playground/data/eval/scienceqa/answers/$CKPT_NAME.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
else
    python -m src.eval.llava.model_vqa_science \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --image-folder $ROOT_DIR/playground//scienceqa/test \
        --question-file $ROOT_DIR/playground/scienceqa/llava_test_CQM-A.json \
        --answers-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}-${IMAGE_SCALE}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale $IMAGE_SCALE
fi

python src/eval/llava/eval_science_qa.py \
    --base-dir $ROOT_DIR/playground/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}-${IMAGE_SCALE}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}-${IMAGE_SCALE}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT_NAME}-${IMAGE_SCALE}_result.json