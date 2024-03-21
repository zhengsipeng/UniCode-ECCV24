#!/bin/bash
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"


CHUNKS=${#GPULIST[@]}

CKPT_PATH=$1
MODEL_BASE=$2
CKPT_NAME=${CKPT_PATH##*/}
CONV_MODE=vicuna_v1


ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava
SPLIT=llava_vqav2_mscoco_test-dev2015


if [ "$CHUNKS" -gt 1 ]; then
    if [ -z "$MODEL_BASE" ]; then
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.llava.model_vqa_loader \
                --model-path $CKPT_PATH \
                --question-file $ROOT_DIR/playground/vqav2/$SPLIT.jsonl \
                --image-folder $ROOT_DIR/playground/vqav2/test2015 \
                --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode $CONV_MODE \
                --image-scale 256 &
        done  
    else
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.llava.model_vqa_loader \
                --model-path $CKPT_PATH \
                --model-base $MODEL_BASE \
                --question-file $ROOT_DIR/playground/vqav2/$SPLIT.jsonl \
                --image-folder $ROOT_DIR/playground/vqav2/test2015 \
                --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
                --num-chunks $CHUNKS \
                --chunk-idx $IDX \
                --temperature 0 \
                --conv-mode $CONV_MODE \
                --image-scale 256 &
        done 
    fi
    
    wait
else
    if [ -z "$MODEL_BASE" ]; then
        python src/eval/llava/model_vqa_loader.py \
            --model-path $CKPT_PATH \
            --question-file $ROOT_DIR/playground/vqav2/$SPLIT.jsonl \
            --image-folder $ROOT_DIR/playground/vqav2/test2015 \
            --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME.jsonl \
            --temperature 0 \
            --conv-mode $CONV_MODE
    else
        python src/eval/llava/model_vqa_loader.py \
            --model-path $CKPT_PATH \
            --model-base $MODEL_BASE \
            --question-file $ROOT_DIR/playground/vqav2/$SPLIT.jsonl \
            --image-folder $ROOT_DIR/playground/vqav2/test2015 \
            --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME.jsonl \
            --temperature 0 \
            --conv-mode $CONV_MODE
    fi
fi

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python src/eval/llava/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT_NAME --dir $ROOT_DIR/playground/vqav2
