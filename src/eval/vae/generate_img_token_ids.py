# This is to generate img token ids for VAE Stage2 Pre-training


# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import os
import logging
import math
import json
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from src.datasets.datasets_vae import ImageNet, CC3M, FFHQ, LLavaDataset
from src.vae.models import build_model
from src.vae.models import ImageGPT2
from src.vae.utils.utils import set_seed
from src.vae.utils.config2 import get_base_config

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dname', type=str, default='imagenet', help='[imagenet, cc3m, ffhq]')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-i', '--input-res', type=int, default=256)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--root-dir', type=str, default="/share/project/datasets/minedojo_april/vlm-pretrain/")
parser.add_argument('--recon-img', type=str, default='all-codes')
parser.add_argument('--code-usage', action='store_true')
parser.add_argument('--fid', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-full-checkpoint', action='store_true')
parser.add_argument('--split', default="val", type=str)

args = parser.parse_args()


def create_dataset(root_dir, name, split="val"):
    transforms_ = [
        transforms.Resize(args.input_res),
        transforms.CenterCrop(args.input_res),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    transforms_ = transforms.Compose(transforms_)

    if name in ['imagenet', 'imgnet']:
        dataset = torchvision.datasets.ImageNet(root=root_dir+'imagenet', split=split, transform=transforms_)
    elif name == 'cc3m':
        dataset = CC3M(root=root_dir+"cc3m", split=split, transform=transforms_)
    elif name == 'ffhq':
        dataset = FFHQ(split=split, transform=transforms_)
    elif name == "llava_laion558k":
        dataset = LLavaDataset(root_dir+'llava/pretrain/BLIP-LAION-CC-SBU-558k/images', split=split, transform=transforms_)
    elif name == "llava_mix665k":
        dataset = LLavaDataset(subset="llava_mix665k", split=split, transform=transforms_)
    else:
        raise ValueError()

    return dataset

 
# for multi-level hqvae
def recon_image(m_codes, model):
    xs_rec = model.decode_code(m_codes)
    xs_rec = torch.clamp(xs_rec, -1., 1.)
    xs_rec = (xs_rec + 1.) / 2.
    return xs_rec


@torch.no_grad()
def do_recon(model, xs, cnts):
    outputs = model(xs)
    xs_rec, codes = outputs[0], outputs[-1]
    xs_rec = torch.clamp(xs_rec, -1., 1.)
    xs_rec = (xs_rec + 1.) / 2.
    xs = (xs + 1.) / 2.
    
    if args.code_usage:
        if isinstance(codes, tuple) or isinstance(codes, list):
            num_code_types = len(cnts)
            for i in range(num_code_types):
                code, cnt = codes[i], cnts[i]
                code, count = torch.unique(code, sorted=True, return_counts=True)
                code = code.view(-1).cpu()
                count = count.view(-1).cpu()
                cnt[code] += count            
        else:
            code, cnt = codes, cnts[0]
            code, count = torch.unique(code, sorted=True, return_counts=True)
            code = code.view(-1).cpu()
            count = count.view(-1).cpu()
            cnt[code] += count

    return xs, xs_rec, codes


@torch.no_grad()
def do_recon_all(model, xs, n_levels):
    if n_levels == 2:
        outputs = model.forward_topbottom(xs)
        xs_rec_all, codes = outputs[0], outputs[-1]

        xs_rec_img = []
        for xs_rec in xs_rec_all:
            xs_rec = torch.clamp(xs_rec, -1., 1.)
            xs_rec = (xs_rec + 1.) / 2.
            xs_rec_img.append(xs_rec)
        xs = (xs + 1.) / 2.

        return xs, xs_rec_img[0], xs_rec_img[1], xs_rec_img[2], codes
    elif n_levels == 3:
        codes = model.get_codes(xs)
        
        # reshape
        B = xs.size(0)
        new_codes = []
        for code in codes:
            K = int(math.sqrt(code.numel()/B))
            code = code.view(B, K, K)
            new_codes.append(code)
        
        codes = new_codes
        xs_rec = recon_image([codes[0], None, None], model)

        xs_rec = torch.clamp(xs_rec, -1., 1.)
        xs_rec = (xs_rec + 1.) / 2.

        xs = (xs + 1.) / 2.  
        return xs, xs_rec, codes


def setup_pretrained_model(model_path):
    config = OmegaConf.load(os.path.join(model_path, "config.yaml"))
    model = build_model(config.stage1.type,
                        config.stage1,
                        config.optimizer)
    last_path = os.path.join(model_path, 'ckpt/last.ckpt')
    ckpt_path = os.path.join(model_path, 'ckpt/state_dict.ckpt')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
    elif os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location='cpu')['state_dict']

    try:
        model.load_state_dict(ckpt, strict=True)
    except RuntimeError:
        print('Changing parameter names for backward compatibility..')
        ckpt_ = {}
        for k, v in ckpt.items():
            if k.startswith('discriminator'):
                ckpt_[k[14:]] = v
            else:
                ckpt_['generator.'+k] = v
        model.load_state_dict(ckpt_, strict=False)
    print(f'{model_path} successfully restored..')
    return model, config


def setup_pretrained_architecture(result_path, device='cuda'):
    config_path = os.path.join(result_path, 'config.yaml')
    last_path = os.path.join(result_path, 'ckpt/last.ckpt')
    ckpt_path = os.path.join(result_path, 'ckpt/state_dict.ckpt')

    config_base = get_base_config(use_default=False)
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config_base, config)
    model = ImageGPT2(config)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)

    elif os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location='cpu')['state_dict']
        model.load_state_dict(ckpt, strict=True)

    config.stage1.hparams_aux.bottom_start = 100000000000 # no bypass 
    model_stage1 = build_model(config.stage1.type,
                               config.stage1,
                               config.optimizer)
    model_stage1.generator.load_state_dict(model.stage1.state_dict())

    return model_stage1, config
    

if __name__ == '__main__':
    set_seed(args.seed)
  
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create console handler and set level to info
    ch = logging.FileHandler(os.path.join(args.result_path, 'eval.log'), mode='a')
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")
    )
    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(logging.StreamHandler())

    dataset = create_dataset(root_dir=args.root_dir, name=args.dname, split=args.split)
    loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                        batch_size=args.batch_size, num_workers=16)
    
    fid = FID().cuda()

    save_path = os.path.join(args.result_path, args.dname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.use_full_checkpoint:
        # for checkpoint with the entire architecture with stage 1 and 2 models
        model, config = setup_pretrained_architecture(args.result_path)
    else:
        # for checkpoint with stage 1 model
        model, config = setup_pretrained_model(args.result_path)     

    model.cuda()
    model.eval()
    
    pbar = tqdm(enumerate(loader), total=len(loader))

    if hasattr(model.generator, 'code_levels'):
        n_levels = model.generator.code_levels
    else:
        n_levels = 2
    cnt_codes = [torch.zeros(config.stage1.n_embed, dtype=torch.int64) for _ in range(n_levels)]  # (code_t, code_b)
    n_samples = 0
    mse_loss = 0

    with open(save_path+"/%s_image_list.json"%args.split, "w") as f:

        if args.dname=="llava_mix665k":
            img_list = []
            for sample in dataset.samples:
                img_list.append(str(sample).replace(args.root_dir+"llava/visual-inst/", ""))    
            json.dump(img_list, f)        
        else:
            json.dump([str(n).split("/")[-1].split(".")[0] for n in dataset.samples], f)
    
    codes_t, codes_b = torch.empty(0, 64).to(model.device), torch.empty(0, 256).to(model.device)

    for it, inputs in pbar:
        xs = inputs[0] if isinstance(inputs, list) else inputs
        xs = xs.cuda()
        
        if args.recon_img == 'top':
            outputs = do_recon_all(model, xs, n_levels)
            xs = outputs[0]
            xs_rec = outputs[1] # top
        else:
            xs, xs_rec, codes = do_recon(model, xs, cnt_codes)

        code_t, code_b = codes[:2]
        # restore_llmcode_id 
        if it==0 and args.split=="train":
            os.makedirs("data/unicode/vae_demo/%s"%args.dname, exist_ok=True)
            for i in range(args.batch_size):
                plt.imsave('data/unicode/vae_demo/%s/image_%d_ori.png'%(args.dname, i), xs[i].permute(1, 2, 0).cpu().numpy())
                plt.imsave('data/unicode/vae_demo/%s/image_%d.png'%(args.dname, i), xs_rec[i].permute(1, 2, 0).cpu().numpy())
                #import pdb;pdb.set_trace()
        code_t = model.generator.quantize_t.selected_llm_indices[code_t].reshape(xs.shape[0], -1)
        code_b = model.generator.quantize_t.selected_llm_indices[code_b].reshape(xs.shape[0], -1)
        
        codes_t = torch.cat([codes_t, code_t])
        codes_b = torch.cat([codes_b, code_b])
        
        mse_loss += F.mse_loss(xs, xs_rec, reduction='sum') / (args.input_res*args.input_res*3)
        n_samples += xs.shape[0]

        pbar.set_description("mse_loss: %.4f" % (mse_loss / n_samples))

    torch.save(codes_t.to(torch.int).cpu(), save_path+"/%s_codes_t.pth"%args.split)
    torch.save(codes_b.to(torch.int).cpu(), save_path+"/%s_codes_b.pth"%args.split)