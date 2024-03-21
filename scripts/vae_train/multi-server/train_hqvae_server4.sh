export PYTHONPATH=.


export NODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.23.4
export MASTER_PORT=23456
export RANK=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0



CKPT_PATH=hqvae_cc3m_bsz448.yaml
CKPT_PATH=hqvae_llava_concat_vqa_bsz384.yaml


python -m torch.distributed.launch \
     --nnodes="${NODES}" \
     --node_rank="${RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
     src/train/train_vae.py -c=configs/unicode/$CKPT_PATH -r=outputs/hqvae --n-gpus=8 --n-nodes=2 
     
#python src/train/train_vae.py -c=configs/unicode/$CKPT_PATH -r=outputs/hqvae --n-gpus=8 --n-nodes=2 