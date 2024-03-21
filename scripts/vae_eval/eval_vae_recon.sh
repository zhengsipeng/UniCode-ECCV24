export PYTHONPATH=.
# Eval ImgNet llmcode 
CUDA_VISIBLE_DEVICES=7 python src/eval/rqvae/compute_vae_rfid.py --split=val \
    --config=configs/pt_origin_norestart.yaml \
    --vqvae=outputs/pt_7b_start0_inc-ratio_1e-2_norestart/23112023_021959/epoch8_model.pt    
