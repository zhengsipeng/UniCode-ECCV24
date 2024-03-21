import os,json
import numpy as np
from glob import glob
from tqdm import tqdm

# read
path = '/share/LLM_project/vlm-pretrain/data/llava/visual-inst'
with open(os.path.join(path,'llava_v1_5_mix665k.json'), 'r', encoding='utf-8') as file:
    image_names = json.load(file)
image_names = [os.path.join(path,each["image"]) for each in tqdm(image_names) if "image" in each]

np.random.shuffle(image_names)
proportion = 0.95
train_names = image_names[:int(len(image_names)*proportion)]
val_names = image_names[int(len(image_names)*proportion):]

# write
with open('/share/LLM_project/vlm-pretrain/LLaVA/src/datasets/assets/pf_train.txt','w') as f:
    for name in tqdm(train_names):
        f.write(name+'\n')
    for line in tqdm(open('/share/LLM_project/vlm-pretrain/LLaVA/src/datasets/assets/pretrain_train.txt', 'r').readlines()):
        f.write(os.path.join('/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/images',line.strip()) +'\n')

with open('/share/LLM_project/vlm-pretrain/LLaVA/src/datasets/assets/pf_val.txt','w') as f:
    for name in tqdm(val_names):
        f.write(name+'\n')
    for line in tqdm(open('/share/LLM_project/vlm-pretrain/LLaVA/src/datasets/assets/pretrain_val.txt', 'r').readlines()):
        f.write(os.path.join('/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/images',line.strip()) +'\n')    
