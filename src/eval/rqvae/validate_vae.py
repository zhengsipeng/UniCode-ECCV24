# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import math
from torch.nn import functional as F
import torch
import torch.distributed as dist

import src.utils.dist as dist_utils
from src.rqvae.models import create_model
from src.datasets import create_dataset
from src.rqvae.optimizer import create_optimizer, create_scheduler
from src.utils.misc import set_seed, compute_model_size, get_num_conv_linear_layers
from src.utils.vae_setup import setup
from src.model import *
from src.train.trainer_rqvae import create_trainer
import glob
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-t', '--test-batch-size', type=int, default=200)
parser.add_argument('-e', '--test-epoch', type=int, default=-1)
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')

args, extra_args = parser.parse_known_args()

set_seed(args.seed)


if __name__ == '__main__':
    config, logger, writer = setup(args, extra_args)
    config.experiment.batch_size = 32
    distenv = config.runtime.distenv
    
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', distenv.local_rank)
    torch.cuda.set_device(device)
    
    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)
    #dataset_val = dataset_trn
    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    
    model = model.to(device)
    if model_ema:
        model_ema = model_ema.to(device)
    steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(
        optimizer, config.optimizer.warmup, steps_per_epoch,
        config.experiment.epochs, distenv
    )

    trainer = create_trainer(config)
    
    train_epochs = config.experiment.epochs
    steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
    epoch_st = 0

    if distenv.master:
        logger.info(f'#conv+linear layers: {get_num_conv_linear_layers(model)}')
    
    assert args.load_path!=""
    disc_state_dict = None
    
    print("Loading from ", args.load_path)
    #
    checkpoint_path = args.load_path+"/epoch1_model.pt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    disc_state_dict = ckpt.get('discriminator', None)
    model = dist_utils.dataparallel_and_sync(distenv, model)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    epoch_st = ckpt['epoch']
    
    llm_codebook = torch.load("outputs/llava_ckpts/llama-2-7b-chat-codebook.pt")
    vae_cb = model.module.quantizer.codebooks[0].weight.data
    sims = F.cosine_similarity(vae_cb[:32000], llm_codebook.data.to(vae_cb.device))
    num = 0
    for i in range(32000):
        if sims[i]>0.50:
            num += 1
            print(num)
    #for i in range(32000):

    import pdb;pdb.set_trace()
    trainer = trainer(model, model_ema, llm_codebook, dataset_trn, dataset_val, config, writer,
                        device, distenv, disc_state_dict=disc_state_dict)
    summary_val = trainer.eval(verbose=True, epoch=epoch_st)

    checkpoint_paths = glob.glob(args.load_path+"/epoch11_model.pt")
    
    checkpoints = sorted(checkpoint_paths)
    
   
    has_inst = False
    for i, checkpoint_path in enumerate(checkpoints):
        #for checkpoint in checkpoints:
        #    if "epoch%d"%i in checkpoint:
        #        checkpoint_path = checkpoint
        #        break
        print(checkpoint_path)
        import pdb;pdb.set_trace()
        torch.cuda.empty_cache()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if hasattr(model, "module"):
            
            new_state_dict = OrderedDict()
            for key, value in ckpt['state_dict'].items():
                new_key = f"module.{key}"  
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
            #model.module.quantizer.codebooks[0].ema
            model.module.quantizer.codebooks[0].embed_ema = llm_codebook
            model.module.quantizer.codebooks[0].weight[:-1, :] = llm_codebook
            #import pdb;pdb.set_trace()
            disc_state_dict = ckpt.get('discriminator', None)
            new_disc_state_dict = OrderedDict()
            for key, value in disc_state_dict.items():
                new_key = f"module.{key}"  
                new_disc_state_dict[new_key] = value
            trainer.discriminator.load_state_dict(new_disc_state_dict)
            
        else:
            model.load_state_dict(ckpt['state_dict'])
            disc_state_dict = ckpt.get('discriminator', None)
            model = dist_utils.dataparallel_and_sync(distenv, model)

        
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch_st = ckpt['epoch']
       
        if not has_inst:
            trainer = trainer(model, model_ema, llm_codebook, dataset_trn, dataset_val, config, writer,
                        device, distenv, disc_state_dict=disc_state_dict)
        has_inst = True
        
        summary_val = trainer.eval(verbose=True, epoch=epoch_st)
        #if trainer.distenv.master:
        #    print("Epoch%d\n"%i)
        #    trainer.logging(summary_val, epoch=i+1, mode='valid')
    #
