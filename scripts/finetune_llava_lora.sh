export PYTHONPATH=.


BASE_CKPT=vicuna-7b-v1.5
INPUT_PRETRAIN_CKPT=llava-v1.5-7b-hqvae-pretrain
OUTPUT_FINETUNE_CKPT=llava-v1.5-7b-hqvae-finetune-lora
#llava_concat_hqvae_epo45+5
VISION_TOWER=llava_laion558k_hqvae_epo25_l2norm

deepspeed --include localhost:0,1,2,3,4,5,6,7 src/train/train_llava.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./configs/zero3.json \
    --model_name_or_path outputs/llava_ckpts/$BASE_CKPT \
    --version v1 \
    --data_path /share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_v1_5_mix665k.json \
    --image_folder /share/LLM_project/vlm-pretrain/data/llava/visual-inst \
    --vision_tower outputs/vae_trained_ckpts/$VISION_TOWER \
    --pretrain_mm_mlp_adapter outputs/$INPUT_PRETRAIN_CKPT/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir outputs/$OUTPUT_FINETUNE_CKPT \
    --logging_dir outputs/$OUTPUT_FINETUNE_CKPT/log \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
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