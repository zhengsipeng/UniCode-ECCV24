import os
from .clip_encoder import CLIPVisionTower
from .vae_encoder import VAEVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    if is_absolute_path_exists and "vae" in vision_tower:
        print("Using VisionTower of %s."%vision_tower)
        return VAEVisionTower(vision_tower, use_quant=vision_tower_cfg.mm_use_quant, **kwargs)

    elif is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')