import os
import torch
import numpy as np
import torch.nn as nn
import yaml
import torchvision.transforms as T
from typing import *
from PIL import Image
from omegaconf import OmegaConf
from src.vae.models import build_model


class VAEVisionTower(nn.Module):
    def __init__(self, vision_tower, use_quant=False, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.use_quant = use_quant
        if not delay_load:
            self.load_vae_model()

    def setup_pretrained_model(self, model_path):
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
    
    def load_model(self):
        self.load_vae_model()

    def load_vae_model(self):
        self.image_processor = None
        print("Loading from {}".format(self.vision_tower_name))
        model, config = self.setup_pretrained_model(self.vision_tower_name) 
        model.eval()
        
        if hasattr(model.generator, "quantize"):
            model.generator.quantize.learnable = False
        else:
            model.generator.quantize_t.learnable = False
            model.generator.quantize_b.learnable = False

        model.requires_grad_(False)
        self.vision_tower = model.generator
        print(dir(self.vision_tower))
        self.is_loaded = True
    
    @property
    def dtype(self):
        return self.vision_tower.encoder.conv_in.weight.dtype

    @property
    def device(self):
        return self.vision_tower.encoder.conv_in.weight.device

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim_output
    
    @torch.no_grad()
    def forward(self, images):
        #import pdb;pdb.set_trace()
        if type(images) is list:
            assert 1==0
        elif self.use_quant:
            code_t, code_b = self.vision_tower.forward_quant(images.to(device=self.device, dtype=self.dtype))
            bsz, dim, _, _ = code_t.shape
            code_t = code_t.reshape(bsz, dim, -1, 1)
            code_b = code_b.reshape(bsz, dim, -1, 4)
            image_features = torch.cat([code_t, code_b], dim=-1).reshape(bsz, dim, -1) # b, dim, 64+256
            image_features = image_features.permute(0, 2, 1)
        else:
            image_quant_outs = self.vision_tower.forward_encode(images.to(device=self.device, dtype=self.dtype)) # b, 8, 8, 256
            
            bsz, h, w, dim = image_quant_outs.shape
            image_features = image_quant_outs.reshape(bsz, h*w, dim)# b, 64, 256

        return image_features