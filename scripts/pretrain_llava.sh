export PYTHONPATH=.


BASE_CKPT=vicuna-7b-v1.5
OUTPUT_PRETRAIN_CKPT=llava-v1.5-7b-hqvae-pretrain
#llava_concat_hqvae_epo45+5
VISION_TOWER=llava_laion558k_hqvae_epo25_l2norm


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29501 src/train/train_llava.py \
    --deepspeed configs/zero2.json \
    --model_name_or_path outputs/llava_ckpts/$BASE_CKPT \
    --version plain \
    --data_path /share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/blip_laion_cc_sbu_558k.json \
    --image_folder /share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/images \
    --vision_tower outputs/vae_trained_ckpts/$VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./outputs/$OUTPUT_PRETRAIN_CKPT \
    --logging_dir ./outputs/$OUTPUT_PRETRAIN_CKPT/log \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
