# UniCode

## 1. Training VAE 

### 1.1 Data
we use images from the following subsets to train the VAE tokenizer of UniCode:

* llava-v1: laion-558K
* llava-v2: (1) laion-558K for LLaVA1.5 pre-training；(2) 300K images from Mixed-665K for LLaVA1.5 finetuning; (3) [TBD] 12 evaluation benchmark of LLaVA1.5

you may refer to [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) to download data, currently all data is organized as follows:
```
llava
-- pretrain
----- BLIP-LAION-CC-SBU-558K
----- LLaVA-CC3M-Pretrain-595K
-- visual-inst
----- coco
----- gqa
----- ocr_vqa
----- textvqa
----- vg
```

we also prepare datasets of VAE benchmarks to sorely train the VAE encoder, aiming to compare it with the original VAE, seeing [DATA_VAE.md](DATA_VAE.md) to download them,
we place these data in "/share/project/datasets/minedojo_april/vlm-pretrain"


### 1.2 Train the VQ-Encoder and Decoder
In this stage, we train the visual tokenizer.
we consider **HQ-VAE** or **RQ-VAE** as the visual quantizer in this project, to pre-train them, run the commands below:


```
# HQ-VAE: output feature maps in 8x8+16x16
sh scripts/vae_train/train_hqvae.sh 

# RQ-VAE: output feature maps in 4x8x8
sh scripts/vae_train/train_rqvae.sh
```

We also provide the multi-server scripts in ``scripts/vae_train/multi-server'', run them separately in each server.

We provide a pretrained hq-vae checkpoint in [llava_concat_hqvae_epo45+5](https://www.dropbox.com/scl/fo/v8a2r608cyj58ayxl1gwq/h?rlkey=x95juv729gtoiraiu7s5c5i18&dl=0).


### 1.3 Train the Code Decoder (not necessary)
we use LLM (e.g., LLaMA-2) to directly decode the image token, this serve as a baseline to compare with our UniCode
```
sh scripts/vae_train/train_hqvae_stage2.sh
```


## 2. Train MLLM
after training the VAE tokenizer, we use a shared codebook to represent both visual and text modalities

### 2.1 Visual Tokenization
First, tokenize the images used for llava pre-training and finetuning
```
bash scripts/vae_eval/generate_img_token_ids.sh $VAE_CKPT_PATH $GPU_ID
```

this will generate visual codes and save them under $VAE_CKPT_PATH

Then,  generate annotations for llava finetuning by running:
```
python data/unicode process.py --func get_t2v_anno
```


To train UniCode for text-only generation, run:
```
bash scripts/finetune_lora.sh
```
We provide a pre-trained UniCode in [unicode-lora-hqvae-concat](https://www.dropbox.com/scl/fo/x5xz94hxm696a96cp0yt7/h?rlkey=rlv3uorj5sbcg52fsgcll53cx&dl=0).

As a comparison, to pre-train and train LLaVA-1.5, run:
```
bash scripts/pretrain_llava.sh
bash scripts/finetune_llava_lora.sh
```


To train UniCode for visual-text generation, run:
```

```


## Evaluation
### Image Reconstruction
```
sh scripts/vae_eval/eval_hqvae.sh 
```
* CKPT_NAME: outputs/xxx
* DATASET: cc3m, imagenet, ffhq, ...
* VAE_TYPE: rqvae, hqvae

### LLaVA Benchmarks
See [Evaluation.md](EVALUATION.md)


### Image Generation