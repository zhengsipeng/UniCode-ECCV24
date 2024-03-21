import json
import os
import glob
from tqdm import tqdm


root_dir = "/share/project/datasets/minedojo_april/vlm-pretrain/llava/visual-inst/ocr_vqa/images"
imgs = glob.glob(root_dir+"/*gif")+glob.glob(root_dir+"/*png")
for img in imgs:
    prefix = img.split(".")[0]
    
    os.system("mv %s %s.jpg"%(img, prefix))
imgs = glob.glob(root_dir+"/*gif")+glob.glob(root_dir+"/*png")
print(len(imgs))
import pdb;pdb.set_trace()


names = ['finetune_val.txt', 'finetune_train.txt', 'pf_val.txt', 'pf_train.txt']
for name in names:
    with open(name) as f:
        lines = [l.strip() for l in f.readlines()]
    new_lines = []
    for l in tqdm(lines):
        new_lines.append(l.replace("/share/LLM_project/vlm-pretrain/data", "/share/project/datasets/minedojo_april/vlm-pretrain"))
    with open(name, "w") as f:
        for l in new_lines:
            f.writelines(l+'\n')
import pdb;pdb.set_trace()

filename = "llava_instruct_150k"
with open(filename+'.json', "r") as f:
    data = json.load(f)

with open(filename+".txt", "w") as f:
    for img_data in data:
        image_name = img_data['image']
        f.writelines(image_name+"\n")
#import pdb;pdb.set_trace()