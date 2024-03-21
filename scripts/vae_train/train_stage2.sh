
export PYTHONPATH=.

OUTPUT_DIR=hqvae_stage2
MODEL_BASE=outputs/llava_ckpts/vicuna-7b-v1.5
META_PATH=/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/llava_laion558k_t2v.json
#llava_laion558k_t2v, blip_laion_cc_sbu_558k.json


deepspeed --include localhost:0 --master_port 29500 src/train/train_unicode.py \
    --deepspeed configs/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --version v1 \
    --data_path $META_PATH \
    --is_unicode True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./outputs/$OUTPUT_DIR \
    --logging_dir ./outputs/$OUTPUT_DIR/log \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard 