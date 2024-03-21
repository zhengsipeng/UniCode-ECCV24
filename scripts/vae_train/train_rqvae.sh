export PYTHONPATH=.

# --resume outputs/hqvae/rqvae_cc3m595k/26122023_101411/ckpt/last.ckpt \

python src/train/train_vae.py \
    -c=configs/unicode/rqvae_cc3m595k.yaml \
    -r=outputs/hqvae --n-gpus=8
