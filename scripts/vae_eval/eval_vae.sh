export PYTHONPATH=.

CKPT_PATH="outputs/hqvae/finished/hqvae_epo25_l2norm"  # rFID: 2.40
#CKPT_PATH="outputs/hqvae/finished/rqvae_epo40+5_l2norm"  #rFID 5.08

DATASET="cc3m595k"


#--code-usage 
CUDA_VISIBLE_DEVICES=1 python src/eval/vae/eval_stage1.py -r=$CKPT_PATH -d=$DATASET --fid