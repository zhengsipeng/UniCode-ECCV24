# UniCode

## Training VAE 

### Data
we use images from the following subsets to train the VAE tokenizer of UniCode:

* llava-v1: laion-558K
* llava-v2: (1) laion-558K for LLaVA1.5 pre-training；(2) 300K images from Mixed-665K for LLaVA1.5 finetuning; (3) [TBD] 12 evaluation benchmark of LLaVA1.5

you may refer to [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) to download data, currently all data is placed in "/share/project/datasets/minedojo_april/vlm-pretrain/llava",
and is organized as follows:
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

we also prepare datasets of VAE benchmarks to sorely train the VAE encoder, aiming to compare it with original VAE, seeing [DATA_VAE.md](DATA_VAE.md) to download them,
we place these data in "/share/project/datasets/minedojo_april/vlm-pretrain"


### Training of Stage 1
we consider **HQ-VAE** or **RQ-VAE** as the visual quantizer, to pre-train them, run the commands below:

HQ-VAE: output feature maps in 8x8+16x16
```
sh scripts/vae_train/train_hqvae.sh
```

RQ-VAE: output feature maps in 4x8x8:
```
sh scripts/vae_train/train_rqvae.sh
```

* CONFIG_NAME: hqvae_llava-v1, hqvae_llava-v2


### Training of Stage 2 (not necessary)
we use LLM (e.g., LLaMA-2) to directly decode the image token
```
sh scripts/vae_train/train_hqvae_stage2.sh
```


## Training VLM
after training the VAE tokenizer, we use a shared codebook to represent both visual and text modalities

As a comparison, to pre-train and train LLaVA-1.5, run:
```

```

To train UniCode for text-only generation, run:
```

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