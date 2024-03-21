export PYTHONPATH=.
# Eval RQ-VAE-8x8x8-llmcode train
#CUDA_VISIBLE_DEVICES=0
MODEL_PATH=outputs/pt_8x8x16-code-fixed-7b-ema-v2-0.99-nocommit/17112023_161733/


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=24641 \
    --nnodes=1 --nproc_per_node=1 --node_rank=0 scripts/eval/rqvae/validate_vae.py \
    -m=$MODEL_PATH/config.yaml \
    -l=$MODEL_PATH \
    -r=$MODEL_PATH/vae_post_valid \
    --split=val 
    