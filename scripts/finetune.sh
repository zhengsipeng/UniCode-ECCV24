export PYTHONPATH=.

CKPT_NAME=$1

# 1M scale 256, vicuna unicode_finetune_lora_norm_1M-vicuna
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29502 src/train/train_unicode.py \
    --deepspeed configs/zero2_offload.json \
    --model_name_or_path outputs/llava_ckpts/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_v1_5_mix665k.json \
    --image_folder /share/LLM_project/vlm-pretrain/data/llava/visual-inst \
    --image_scale 256 \
    --vision_tower outputs/vae_trained_ckpts/llava_laion558k_hqvae_epo35+5 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_quant True \
    --mm_use_mlp False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./outputs/$CKPT_NAME \
    --logging_dir ./outputs/$CKPT_NAME/log \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
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