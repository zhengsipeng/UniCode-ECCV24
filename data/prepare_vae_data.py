import os
from tqdm import tqdm


def prepare_cc3m(root_dir):
    # Filter out images that do not exists in the folder (e.g., fail download)
    with open(os.path.join(root_dir, "val_list_all.txt")) as f:
        val_images = [l for l in f.readlines()]
    
    _val_images = []
    for val_img in tqdm(val_images):
        if os.path.exists(root_dir+"/"+val_img.split("\t")[0]):
            _val_images.append(val_img)
    print("Val images number after cleaning: %d"%len(_val_images))
    
    with open(os.path.join(root_dir, "val_list.txt"), "w") as f:
        for val_img in _val_images:
            f.writelines(val_img)


    with open(os.path.join(root_dir, "train_list_all.txt")) as f:
        val_images = [l for l in f.readlines()]
    
    _val_images = []
    for val_img in tqdm(val_images):
        if os.path.exists(root_dir+"/"+val_img.split("\t")[0]):
            _val_images.append(val_img)
            if len(_val_images)%10000==0:
                print(len(_val_images))
    print("Train images number after cleaning: %d"%len(_val_images))
    
    with open(os.path.join(root_dir, "train_list.txt"), "w") as f:
        for val_img in _val_images:
            f.writelines(val_img)
    #import pdb;pdb.set_trace()



    prepare_cc3m(root_dir="/share/LLM_project/vlm-pretrain/data/cc3m")