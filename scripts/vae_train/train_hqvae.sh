export PYTHONPATH=.


CKPT_PATH=hqvae_llava_concat_bsz384.yaml


python src/train/train_vae.py -c=configs/$CKPT_PATH -r=outputs/hqvae --n-gpus=8


