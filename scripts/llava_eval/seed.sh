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
CONV_MODE=vicuna_v1

if [ -z "$MODEL_BASE" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.llava.model_vqa_loader \
            --model-path $CKPT_PATH \
            --question-file $ROOT_DIR/playground/seed-bench/llava-seed-bench_1.jsonl \
            --image-folder $ROOT_DIR/playground/seed-bench \
            --answers-file ./playground/data/eval/seed_bench/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
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
            --question-file $ROOT_DIR/playground/seed-bench/llava-seed-bench_1.jsonl \
            --image-folder $ROOT_DIR/playground/seed-bench \
            --answers-file ./playground/data/eval/seed_bench/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode $CONV_MODE \
            --image-scale $IMAGE_SCALE &
    done
fi
wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


# Evaluate
python src/eval/llava/convert_seed_for_submission.py \
    --annotation-file $ROOT_DIR/playground/seed-bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT_NAME.jsonl