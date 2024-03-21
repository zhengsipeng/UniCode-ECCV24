import os
import torch
import json
import pandas as pd
import random
import base64
import argparse
from io import BytesIO
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--func', default="", type=str)
args = parser.parse_args()


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


generate_prompts = [
    "Generate an image based on this description: '{}'.",
    "Create a photo from the following description: '{}'.",
    "Produce an illustration that reflects this summary: '{}'.",
    "Formulate an image in accordance with the details provided: '{}'.",
    "Develop a picture of the following description: '{}'.",
    "Craft an image according to this narrative: '{}'.",
    "Compose a picture based on the summary: '{}'.",
    "Output an image aligned with this content summary: '{}'.",
    "Render a photo following this brief description: '{}'.",
    "Construct an image that corresponds to the following details: '{}'.",
    "Design an image that matches this description: '{}'.",
    "Generate an image that encapsulates this description: '{}'.",
    "Output an image reflecting this description: '{}'.",
    "Translate the following description into an image: '{}'.",
    "Produce a photo from this visual caption: '{}'.",
    "Sketch out an image adhering to this description: '{}'."
]

generate_partial_prompts = [
    "Generate the next part of a partial image with the description of '{}', conditioned on given image tokens: {}.",
    "Create the subsequent section of a partially completed image, as per the description '{}', using the provided image tokens: {}.",
    "Produce the following segment of an incomplete image, described as '{}', based on the given image tokens: {}.",
    "Develop the next portion of a partial photo summarized as '{}', with the image tokens: {}.",
    "Formulate the ensuing part of an image that's not yet complete, guided by '{}', and using the following visual tokens: {}.",
    "Construct the next region of a partially finished image of '{}', based on the provided image tokens: {}.",
    "Compose the subsequent area of an incomplete image, described by '{}', conditioned on the given tokens: {}.",
    "Craft the ensuing section of a partial photo of '{}', utilizing its given tokens: {}.",
    "Generate the subsequent fragment of an incomplete image, described as '{}', using the specified image tokens: {}.",
    "Craft the next part of a partial picture, following the narrative '{}' and conditioned on the image tokens: {}.",
    "Create the following slice of a partial image, as per '{}', conditioned on the image tokens: {}.",
    "Produce the next part of a picture in progress described by '{}', using its available image tokens: {}.",
    "Render the subsequent portion of a partially completed image conditioned on its summary '{}' and given image tokens: {}.",
    "Develop the following region of an image that's not fully completed, as described in '{}', based on its partial visual tokens: {}.",
    "Formulate the next segment of a partial image with the narrative '{}', using its known tokens: {}.",
    "Output the ensuing part of an image still under creation, based on its caption '{}' and known tokens: {}.",
    "Generate the subsequent section of a partially finished image following '{}', conditioned on the image tokens: {}.",
    "Forge the next component of a partially completed image as outlined in '{}', while utilizing its specified image tokens: {}."
]


def move_cc12m_imgs():
    cc12m_dir = "/share/project/datasets/minedojo_april/vlm-pretrain/cc12m"

    imgs = os.listdir(cc12m_dir+"/images4_resize")
    for img in tqdm(imgs):
        #import pdb;pdb.set_trace()
        os.system("mv %s/images4_resize/%s %s/images"%(cc12m_dir, img, cc12m_dir))


def codes_loader(vae_ckpt_dir):
    img_list = []
    with open(vae_ckpt_dir+"/train_image_list.json") as f:
        img_list += json.load(f)
   
    with open(vae_ckpt_dir+"/val_image_list.json") as f:
        img_list += json.load(f)

    print("Loading codes")
    trn_codes_t = torch.load(vae_ckpt_dir+"/train_codes_t.pth")
    trn_codes_b = torch.load(vae_ckpt_dir+"/train_codes_b.pth")
    val_codes_t = torch.load(vae_ckpt_dir+"/val_codes_t.pth")
    val_codes_b = torch.load(vae_ckpt_dir+"/val_codes_b.pth")
    codes_t = torch.cat([trn_codes_t, val_codes_t])
    codes_b = torch.cat([trn_codes_b, val_codes_b])
    
    img_name2id = dict( zip(img_list, range(len(img_list))) )
   
    return img_name2id, codes_t, codes_b


def get_llava_uni_anno(anno_path, vae_ckpt_dir):
    if "blip" in anno_path:
        dataset = "llava_laion558k"
    else:
        dataset = "llava_mix665k"
        
    with open(anno_path) as f:
        data = json.load(f)
    
    img_name2id, codes_t, codes_b = codes_loader(vae_ckpt_dir+"/"+dataset)
    imgs_id = []
    new_data= []
    
    for img_inst in tqdm(data):
        
        if "image" not in img_inst:
            new_data.append(img_inst)
        else:
            img_id = img_name2id[img_inst['image']]

            img_codes_t = codes_t[img_id].unsqueeze(1)
            img_codes_b = codes_b[img_id].reshape(64, 4)
            img_codes = torch.cat([img_codes_t, img_codes_b], dim=-1).flatten()
            img_codes = img_codes.tolist()

            new_data.append({"id": img_inst['image'], 'visual_tokens': img_codes, 'conversations': img_inst['conversations']})

    anno_file = anno_path.split("/")[-1]
    print(len(new_data))
    with open(os.path.join(anno_path.replace(anno_file, ""), "%s.json"%dataset), "w") as f:
        json.dump(new_data, f)


def get_t2v_anno(root_dir, vae_ckpt_dir):
    gen_type = ['full', 'part']
    with open(os.path.join(root_dir, "blip_laion_cc_sbu_558k.json")) as f:
        data = json.load(f)
    
    img_name2id, codes_t, codes_b = codes_loader(vae_ckpt_dir)
    
    imgs_id = []
    full_data, part_data = [], []
    for img_inst in tqdm(data):
        img_id = img_name2id[img_inst['id']]
        if img_id not in imgs_id:
            imgs_id.append(img_id)

        img_codes_t = codes_t[img_id].unsqueeze(1)
        img_codes_b = codes_b[img_id].reshape(64, 4)
        img_codes = torch.cat([img_codes_t, img_codes_b], dim=-1).flatten()
        img_codes = img_codes.tolist()
        
        imgid, imgname, conv = img_inst['id'], img_inst['image'], img_inst['conversations']
        desc = conv[1]['value']
        assert len(conv)==2

        # full
        query_prompt = random.choice(generate_prompts)
        
        query = query_prompt.format(desc)
        answer = "<vistok>"+"-".join([str(d) for d in img_codes])+"</vistok>."
        full_data.append({'id': imgid, 'conversations': [{'from': 'human', 'value': query},
                                                         {'from': 'gpt',   'value': answer}]})
 
        # partial, 8x(4+1)=40
        part = random.randint(0, 6)
        query_prompt = random.choice(generate_partial_prompts)
        
        query_codes = img_codes[:part*40]
        answer_codes = img_codes[part*40:(part+1)*40]
        
        query_codes = "<vistok>"+"-".join([str(d) for d in query_codes])+"</vistok>"
        answer_codes = "<vistok>"+"-".join([str(d) for d in answer_codes])+"</vistok>"
       
        query = query_prompt.format(desc, query_codes)
        
        answer = answer_codes + "."

        part_data.append({'id': imgid, 'conversations': [{'from': 'human', 'value': query},
                                                        {'from': 'gpt',   'value': answer}]})
        
  
    all_data = full_data + part_data  
    random.shuffle(full_data)
    random.shuffle(part_data)
    random.shuffle(all_data)
    
    print("number of imgids", len(imgs_id)) 
    print("number of samples", len(all_data))
    with open(os.path.join(root_dir, "llava_laion558k_t2v.json"), "w") as f:
        json.dump(all_data, f)
    with open(os.path.join(root_dir, "llava_laion558k_t2v_full.json"), "w") as f:
        json.dump(full_data, f)
    with open(os.path.join(root_dir, "llava_laion558k_t2v_part.json"), "w") as f:
        json.dump(part_data, f)


def concat_llava_anno():
    print("Loading anno1")
    with open("/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/blip_laion_cc_sbu_558k.json") as f:
        anno_1 = json.load(f)
        
    anno = []
    for sample in anno_1:
        
        sample['split'] = 0
        anno.append(sample)
        
    print("Loading anno2")
    with open("/share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_v1_5_mix665k.json") as f:
        anno_2 = json.load(f)
    
    for sample in anno_2:
        sample['split'] = 1
        anno.append(sample)

    random.shuffle(anno)
    print(len(anno))

    with open("/share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_v1_5_concat.json", "w") as f:
        json.dump(anno, f)
        

def concat_uni_anno():
    print("Loading anno1")
    with open("/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/llava_laion558k_imgen.json") as f:
        anno_1 = json.load(f)
        
    anno = []
    for sample in anno_1:
        import pdb;pdb.set_trace()
        sample['split'] = 0
        anno.append(sample)
        
    print("Loading anno2")
    with open("/share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_mix665k_imgen.json") as f:
        anno_2 = json.load(f)
    
    for sample in anno_2:
        sample['split'] = 1
        anno.append(sample)

    random.shuffle(anno)
    print(len(anno))
    #import pdb;pdb.set_trace()
    with open("/share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_laion558k_mix665k.json", "w") as f:
        json.dump(anno, f)
        

def concat_vae_with_mix():
    root_dir = '../../src/datasets/assets/'
    with open(root_dir+"llava_laion558k_train.txt") as f:
        imgs =  [l.strip() for l in f.readlines()]
    with open(root_dir+"llava_laion558k_test.txt") as f:
        imgs +=  [l.strip() for l in f.readlines()]
    with open(root_dir+"llava_mix665k_train.txt") as f:
        imgs +=  [l.strip() for l in f.readlines()]
    with open(root_dir+"llava_mix665k_val.txt") as f:
        imgs +=  [l.strip() for l in f.readlines()]
    random.shuffle(imgs)
    with open(root_dir+"llava_concat_train.txt", "w") as f:
        for img in imgs:
            f.writelines(img+"\n")
    print(len(imgs))


def concat_vae_with_vqabench():
    root_dir = '../../src/datasets/assets/'
    with open(root_dir+"llava_concat_train.txt") as f:
        imgs = [l.strip() for l in f.readlines()]

    img_dict = {}
    for img in imgs:
        img_dict[img.split("/")[-1].split(".")[0]] = img
    
    vqa_root_path = "/share/LLM_project/vlm-pretrain/data/llava/"

    num_vqa_imgs = 0
    for benchmark in ['llavabench', 'mm-vet', 'seed-bench', 'pope', 'textvqa', 'scienceqa', 'vizwiz', 'vqav2', 'mme', 'mmbench', 'mmbench_cn', 'gqa']:
        print("Starting %s"%benchmark)
        if benchmark in ['llavabench', 'mm-vet', 'seed-bench', 'pope', 'textvqa', 'vizwiz', 'vqav2', 'mme', "gqa"]:
            if benchmark=="gqa":
                question_file = vqa_root_path+"playground/gqa/llava_gqa_testdev_balanced.jsonl"
            elif benchmark=="mme":
                question_file = vqa_root_path+"playground/MME/llava_mme.jsonl"
            elif benchmark=='vqav2':
                question_file = vqa_root_path+"playground/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl"
            elif benchmark=='vizwiz':
                question_file = vqa_root_path+"playground/vizwiz/llava_test.jsonl"
            elif benchmark=='textvqa':
                question_file = vqa_root_path+"playground/textvqa/llava_textvqa_val_v051_ocr.jsonl"
            elif benchmark=='pope':
                question_file = vqa_root_path+"playground/pope/llava_pope_test.jsonl"
            elif benchmark=='seed-bench':
                question_file = vqa_root_path+"playground/seed-bench/llava-seed-bench_1.jsonl"
            elif benchmark=='mm-vet':
                question_file = vqa_root_path+"playground/mm-vet/llava-mm-vet.jsonl"
            elif benchmark=='llavabench':
                question_file = vqa_root_path+"playground/llava-bench-in-the-wild/questions.jsonl"

            questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
            for q in tqdm(questions):
                img_id = q['image'].split(".")[0]

                if img_id not in img_dict:
                    if benchmark=='gqa':
                        img_path = vqa_root_path+"visual-inst/gqa/images/"+q['image']
                    elif benchmark=='mme':
                        img_path = vqa_root_path+"playground/MME/MME_Benchmark_release_version/%s"%q['image']
                    elif benchmark=='vqav2':
                        img_path = vqa_root_path+"playground/vqav2/test2015/%s"%q['image']
                    elif benchmark=="vizwiz":
                        img_path = vqa_root_path+"playground/vizwiz/test/%s"%q['image']
                    elif benchmark=='textvqa':
                        img_path = vqa_root_path+"playground/textvqa/train_images/%s"%q['image']
                    elif benchmark=='pope':
                        img_path = vqa_root_path+"playground/pope/val2014/%s"%q['image']
                    elif benchmark=='seed-bench':
                        img_path = vqa_root_path+"playground/seed-bench/%s"%q['image']
                    elif benchmark=='mm-vet':
                        img_path = vqa_root_path+"playground/mm-vet/images/%s"%q['image']
                    elif benchmark=='llavabench':
                        img_path = vqa_root_path+"playground/llava-bench-in-the-wild/images/%s"%q['image']
                       
                    assert os.path.exists(img_path)
                    #import pdb;pdb.set_trace()
                    imgs.append(img_path)
                    img_dict[img_id] = img_path
                    num_vqa_imgs += 1
        elif benchmark=="scienceqa":
            question_file = vqa_root_path+"playground/scienceqa/llava_test_CQM-A.json"
            with open(question_file) as f:
                questions = json.load(f)
            for q in tqdm(questions):
                if 'image' in q:
                    img_path = vqa_root_path+"playground/scienceqa/test/%s"%q['image']
                    assert os.path.exists(img_path)
                    imgs.append(img_path)
                    num_vqa_imgs += 1
              
        elif benchmark in ["mmbench", "mmbench_cn"]:
            if benchmark=="mmbench":
                questions = pd.read_table(vqa_root_path+"playground/mmbench/mmbench_dev_20230712.tsv")
            else:
                questions = pd.read_table(vqa_root_path+"playground/mmbench_cn/mmbench_dev_cn_20231003.tsv")

            for index, row in tqdm(questions.iterrows(), total=len(questions)):
                idx = row['index']
                question = row['question']
                hint = row['hint']
                image = load_image_from_base64(row['image'])
                os.makedirs(vqa_root_path+"playground/%s/images"%benchmark, exist_ok=True)
                img_path = vqa_root_path+"playground/%s/images/%s-%d.jpg"%(benchmark, benchmark, idx)
                if not os.path.exists(img_path):
                    image.save(img_path)
                imgs.append(img_path)
                num_vqa_imgs += 1

    print("VQA %.2fK"%(num_vqa_imgs/1000))
    print("ALL %.2fK"%(len(imgs)/1000))
    random.shuffle(imgs)
    with open(root_dir+"llava_concat_vqa_train.txt", "w") as f:
        for img in imgs:
            f.writelines(img+"\n")
    os.system("cp %s/llava_mix665k_val.txt %s/llava_concat_vqa_val.txt"%(root_dir, root_dir))


def rename_llava_mix_list():
    with open("../../src/datasets/assets/llava_mix665k_train.txt") as f:
        imgs = [l.strip() for l in f.readlines()]
    
    new_imgs = []
    for img in imgs:
        img = img.replace("/share/project/datasets/minedojo_april/vlm-pretrain", "/share/LLM_project/vlm-pretrain/data")
        new_imgs.append(img)
    with open("../../src/datasets/assets/llava_mix665k_train_2.txt", "w") as f:
        for img in new_imgs:
            f.writelines(img+"\n")
            #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
            
            
if __name__=="__main__":
    root_dir = "/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k"
    anno_path = "/share/LLM_project/vlm-pretrain/data/llava/pretrain/BLIP-LAION-CC-SBU-558k/blip_laion_cc_sbu_558k.json"
    anno_path = "/share/LLM_project/vlm-pretrain/data/llava/visual-inst/llava_v1_5_mix665k.json"
   
    vae_ckpt_dir = "../../outputs/hqvae/finished/llava_laion558k_hqvae_epo35+5"
    
    if args.func =='get_t2v_anno' 
        get_t2v_anno(root_dir, dataset, vae_ckpt_dir)  # to generate llava-like metadata for image generation
    elif args.func == 'get_llava_uni_anno'
        get_llava_uni_anno(anno_path, vae_ckpt_dir)
    elif args.func == 'concat_uni_anno':
        concat_uni_anno()
    elif args.func == 'concat_llava_anno':
        concat_llava_anno()  # to concatenate metadata of laion558k and mix665k for vqa benchmarks
    elif args.func == 'concat_vae_with_mix':
        concat_vae_with_mix()
    elif args.func == 'concat_vae_with_vqabench':
        concat_vae_with_vqabench()