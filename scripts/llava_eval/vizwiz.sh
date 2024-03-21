export PYTHONPATH=.


CKPT_PATH=$1
MODEL_BASE=$2
IMAGE_SCALE=$3

CKPT_NAME=${CKPT_PATH##*/}
ROOT_DIR=/share/LLM_project/vlm-pretrain/data/llava

CONV_MODE=vicuna_v1
#IMG_SCALE=448


# finetune by 336, rescale to 336
if [ -z "$MODEL_BASE" ]; then
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file $ROOT_DIR/playground/vizwiz/llava_test.jsonl \
        --image-folder $ROOT_DIR/playground/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale 256
else
    python -m src.eval.llava.model_vqa_loader \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file $ROOT_DIR/playground/vizwiz/llava_test.jsonl \
        --image-folder $ROOT_DIR/playground/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/$CKPT_NAME.jsonl \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --image-scale 256
fi


python src/eval/llava/convert_vizwiz_for_submission.py \
    --annotation-file $ROOT_DIR/playground/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT_NAME}.json
