#!/bin/bash
export PYTHONPATH=.

CKPT_PATH=$1
MODEL_BASE=$2
CKPT_NAME=${CKPT_PATH##*/}

ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
CONV_MODE=vicuna_v1

if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/llava-bench-in-the-wild/questions.jsonl \
        --image-folder $ROOT_DIR/playground/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE
else
        python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/llava-bench-in-the-wild/questions.jsonl \
        --image-folder $ROOT_DIR/playground/llava-bench-in-the-wild/images \
        --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE
fi

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews


python src/eval/llava/eval_gpt_review_bench.py \
    --question $ROOT_DIR/playground/llava-bench-in-the-wild/questions.jsonl \
    --context $ROOT_DIR/playground/llava-bench-in-the-wild/context.jsonl \
    --rule src/eval/llava/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$CKPT_NAME.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT_NAME.jsonl



python src/eval/llava/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$CKPT_NAME.jsonl