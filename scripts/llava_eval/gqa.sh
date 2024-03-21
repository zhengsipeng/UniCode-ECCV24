#!/bin/bash
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3

CKPT_NAME=${CKPT_PATH##*/}
ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
SPLIT=llava_gqa_testdev_balanced
GQADIR=./playground/data/eval/gqa
CONV_MODE=vicuna_v1

#: '
if [ "$CHUNKS" -gt 1 ]; then
    if [ -z "$MODEL_BASE" ]; then
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.llava.model_vqa_loader \
                --model-path $CKPT_PATH \
                --image-folder $ROOT_DIR/visual-inst/gqa/images \
                --question-file $ROOT_DIR/playground/gqa/$SPLIT.jsonl \
                --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode $CONV_MODE \
                --image-scale $IMAGE_SCALE &
        done
    else
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.llava.model_vqa_loader \
                --model-path $CKPT_PATH \
                --model-base $MODEL_BASE \
                --image-folder $ROOT_DIR/visual-inst/gqa/images \
                --question-file $ROOT_DIR/playground/gqa/$SPLIT.jsonl \
                --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode $CONV_MODE \
                --image-scale $IMAGE_SCALE &
        done
    fi 
    wait
else
    if [ -z "$MODEL_BASE" ]; then
        python src/eval/llava/model_vqa_loader.py \
            --model-path $CKPT_PATH \
            --image-folder $ROOT_DIR/visual-inst/gqa/images \
            --question-file $ROOT_DIR/playground/gqa/$SPLIT.jsonl \
            --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME.jsonl \
            --temperature 0 \
            --conv-mode $CONV_MODE
    else
        python src/eval/llava/model_vqa_loader.py \
            --model-path $CKPT_PATH \
            --model-base $MODEL_BASE \
            --image-folder $ROOT_DIR/visual-inst/gqa/images \
            --question-file $ROOT_DIR/playground/gqa/$SPLIT.jsonl \
            --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME.jsonl \
            --temperature 0 \
            --conv-mode $CONV_MODE
    fi
fi
#'

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python src/eval/llava/convert_gqa_for_eval.py --src $output_file --dst $ROOT_DIR/playground/gqa/testdev_balanced_predictions.json


python src/eval/llava/eval_gqa.py --tier testdev_balanced --dir $ROOT_DIR/playground/gqa
