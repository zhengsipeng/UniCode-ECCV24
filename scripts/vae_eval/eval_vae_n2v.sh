export PYTHONPATH=.

: '


PORT=26553
RQVAE_CKPT=outputs/rqvae_ckpts/cat/stage1/model.pt
RQTRANSFORMER_CKPT=outputs/rqvae_ckpts/cat/stage2/model.pt
SAVE_IMG_DIR=outputs/LSUN-cat-val


PORT=26552
RQVAE_CKPT=outputs/rqvae_ckpts/bedroom/stage1/model.pt
RQTRANSFORMER_CKPT=outputs/rqvae_ckpts/bedroom/stage2/model.pt
SAVE_IMG_DIR=outputs/LSUN-bedroom-val

PORT=26551
RQVAE_CKPT=outputs/rqvae_ckpts/church/stage1/model.pt
RQTRANSFORMER_CKPT=outputs/rqvae_ckpts/church/stage2/model.pt
SAVE_IMG_DIR=outputs/LSUN-church-val
'

PORT=26551
RQVAE_CKPT=outputs/rqvae_ckpts/imagenet_1.4B/stage1/model.pt
RQTRANSFORMER_CKPT=outputs/rqvae_ckpts/imagenet_1.4B/stage2/model.pt
SAVE_IMG_DIR=outputs/ImageNet-val

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port=$PORT \
  --nnodes=1 --nproc_per_node=1 --node_rank=0 src/eval/rqvae/main_sampling_fid.py \
  -v=$RQVAE_CKPT -a=$RQTRANSFORMER_CKPT --save-dir=$SAVE_IMG_DIR