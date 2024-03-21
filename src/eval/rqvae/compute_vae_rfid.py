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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os
import torch
import warnings
warnings.filterwarnings("ignore")
from src.datasets import create_dataset
from src.rqvae.models import create_model
from src.rqvae.metrics.fid import compute_rfid
from src.utils.vae_config import load_config, augment_arch_defaults
from src.model import *


def load_model(path, ema=False, config=None):    
    if config is None:
        model_config = os.path.join(os.path.dirname(path), 'config.yaml')
        config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)
    #
    model, _ = create_model(config.arch, ema=False)
    ckpt = torch.load(path)['state_dict_ema'] if ema else torch.load(path)['state_dict']
    model.load_state_dict(ckpt)
    
    return model, config


def setup_logger(result_path):
    log_fname = os.path.join(result_path, 'rfid.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == '__main__':
    """
    Computes rFID, i.e., FID between val images and reconstructed images.
    Log is saved to `rfid.log` in the same directory as the given vqvae model. 
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=300,
                        help='Batch size to use')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--vqvae', type=str, default='', required=True,
                        help='vqvae path for recon FID')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    
        
    result_path = os.path.dirname(args.vqvae)
    logger = setup_logger(result_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    config = load_config(args.config) if args.config is not None else None

    vqvae_model, config = load_model(args.vqvae, config=config)
    
    if config.arch.hparams.bottleneck_type=='llm':
        compute_dtype = (torch.float16 if config.llama.fp16 else (torch.bfloat16 if config.llama.bf16 else torch.float32))
        bnb_model_from_pretrained_args = {}
        if config.llama.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(dict(
                #device_map={"": training_args.device},
                device_map='auto',
                load_in_4bit=config.llama.bits == 4,
                load_in_8bit=config.llama.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=config.llama.bits == 4,
                    load_in_8bit=config.llama.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    torch_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=config.llama.double_quant,
                    bnb_4bit_quant_type=config.llama.quant_type # {'fp4', 'nf4'}
                )
            ))
            
        vqvae_model.quantizer = QuantForCausalLM.from_pretrained(
            config.llama.model_name_or_path,
            cache_dir=config.llama.cache_dir,
            **bnb_model_from_pretrained_args
        )
        vqvae_model.quantizer.init(config.arch.hparams)
        vqvae_model.quantizer.requires_grad_(False)
        vqvae_model.quantizer.to(torch.bfloat16)
        vqvae_model.quantizer.config.use_cache = False
             
    if args.root is not None:
        config.dataset.root = args.root
    
    vqvae_model = vqvae_model.to(device)
    vqvae_model = torch.nn.DataParallel(vqvae_model).eval()
    logger.info(f'vqvae model loaded from {args.vqvae}')
    
    dataset_trn, dataset_val = create_dataset(config, is_eval=True, logger=logger)
    if config.dataset.type in ['LSUN-church', "LSUN-bedroom"]:
        dataset_val = dataset_trn
    
    dataset = dataset_val if args.split in ['val', 'valid'] else dataset_trn
    logger.info(f'measuring rFID on {config.dataset.type}/{args.split}')
    
    rfid = compute_rfid(dataset, vqvae_model, batch_size=args.batch_size, device=device)
    logger.info(f'rFID: {rfid:.4f}')
