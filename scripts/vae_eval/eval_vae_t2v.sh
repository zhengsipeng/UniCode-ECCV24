export PYTHONPATH=.

PORT=23566
RQVAE_CKPT=outputs/rqvae_ckpts/cc3m/stage1/model.pt
RQTRANSFORMER_CKPT=outputs/rqvae_ckpts/cc3m/stage2/model.pt
SAVE_IMG_DIR=outputs/cc3m_val


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --master_port=$PORT \
  --nnodes=1 --nproc_per_node=1 --node_rank=0 src/eval/vae/main_sampling_txt2img.py \
  -v=$RQVAE_CKPT -a=$RQTRANSFORMER_CKPT --dataset="cc3m" --save-dir=$SAVE_IMG_DIR