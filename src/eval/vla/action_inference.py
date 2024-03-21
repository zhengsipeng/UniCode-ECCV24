import argparse
import torch
import numpy as np

from src.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN
from src.utils.conversation import conv_templates, SeparatorStyle
from src.model.builder import load_pretrained_model
from src.utils.misc import disable_torch_init
from src.utils.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from src.utils.vla_utils import decode_action, decode_inst
from src.utils.vla_utils import NUM_BINS, ACTION_TOKEN_BIAS, ACTION_DIM, TD_BINS
from src.datasets.rtx_utils import resize_to_resolution, DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS, get_steps_dataset
from src.datasets.rtx_utils import load_data_from_group
import matplotlib.pyplot as plt

import os
from PIL import Image

import tensorflow_datasets as tfds
import tensorflow as tf
import rlds
import h5py

class VLAPolicy:
    
    def __init__(self, model_path, model_base, output_layer, conv_mode='vla', embodiment = 'language_table_sim'):
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.embodiment = embodiment
        
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name)
        
        self.model.config.addition_output_layer_k = output_layer

        if embodiment == 'language_table' or embodiment == 'language_table_sim':
            self.allowed_tokens = list(self.tokenizer.get_vocab().keys())[ACTION_TOKEN_BIAS:TD_BINS+ACTION_TOKEN_BIAS]
        else:
            self.allowed_tokens = list(self.tokenizer.get_vocab().keys())[ACTION_TOKEN_BIAS:NUM_BINS+ACTION_TOKEN_BIAS]
        self.allowed_token_ids = self.tokenizer.convert_tokens_to_ids(self.allowed_tokens)

    def generate_action(self, input_image: np.ndarray, instruction: str, embodiment: str = None):
        if not embodiment:
            embodiment = self.embodiment
        
        qs = instruction
        qs = 'What action should the robot take to ' + qs + ' ?'
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[self.conv_mode].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # should be tested
        prompt += (" " + DEFAULT_ACT_START_TOKEN)
        #print(input_image)
        input_image = input_image.astype(np.uint8)
        image = Image.fromarray(input_image).convert('RGB')

        #image.save("play_lts.jpg")
        #import pdb;pdb.set_trace()
        if embodiment == 'maniskill':
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt', do_normalize=True, image_mean=[0.30663194,0.33064313,0.37233604], image_std=[0.1308936,0.12518778,0.11768314])['pixel_values'].half().cuda()
        else:
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        #import pdb;pdb.set_trace()
        
        #print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        if embodiment == 'language_table' or embodiment == 'language_table_sim':
            generation_new_tokens = 2
        else:
            generation_new_tokens = 7

        with torch.inference_mode():
            #import pdb;pdb.set_trace()
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=generation_new_tokens,
                min_new_tokens=generation_new_tokens,
                use_cache=True,
                prefix_allowed_tokens_fn=lambda b, i: self.allowed_token_ids)
            
        input_token_len = input_ids.shape[1]
        #print(output_ids[:, input_token_len:])
        #outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        #outputs = outputs.strip()
        #print(outputs)
        #if outputs.endswith(stop_str):
        #    outputs = outputs[:-len(stop_str)]
        #outputs = outputs.strip()
        #output_tokens = self.tokenizer.tokenize(outputs)
        #output_tokens = outputs.split()

        #action_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        action_ids = output_ids[:, input_token_len:][0]
        print(action_ids)
        action = decode_action(action_ids, self.tokenizer, embodiment)
        return action
    
if __name__ == '__main__':
    #model_path = 'outputs/vla_ckpts/vla_vicuna_lora_mmload_maniskill_0307_2layers_32batch'
    #model_path = 'outputs/vla_ckpts/vla_vicuna_lora_mmload_maniskill_pickup_0229_4layers_32batch'
    model_path = 'outputs/vla_ckpts/vla_vicuna_lora_mmload_languageTSim_0319_16layers_64batch'
    model_base = 'outputs/llava_ckpts/vicuna-7b-v1.5'
    
    #dataset_builder = tfds.builder_from_directory(builder_dir='/share/LLM_project/vlm-pretrain/unicode/data/rtx/franka/maniskill_dataset_converted_externally_to_rlds/0.1.0')
    
    #ds = dataset_builder.as_dataset(split='train')

    #v = DATASET_NAME_TO_TRAJECTORY_DATASET_KWARGS['taco_play']
    #dataset = get_steps_dataset(**v)
    #steps = dataset

    #ds.shuffle(100)
    #ds_iter = iter(ds)
    #episode = next(ds_iter)
    #episode = next(ds_iter)
    #episode = next(ds_iter)
    #steps = episode[rlds.STEPS]

    policy = VLAPolicy(model_path, model_base, output_layer=16, embodiment='language_table_sim')

    data_root_path = '/share/LLM_project/vlm-pretrain/unicode/data/language_table_sim_hdf5'
    h5_file_path = os.path.join(data_root_path, "step_dataset_0.h5")
    
    prediction_x = []
    prediction_y = []
    gt_x = []
    gt_y = []
    
    with h5py.File(h5_file_path, 'r') as hf:
        for index in range(20):
            sample_group = hf['train'][str(index)]
            sample_data = load_data_from_group(sample_group)

            im = sample_data['observation']['image']
            inst = decode_inst(sample_data['observation']['language_instruction'])
            prediction = policy.generate_action(im, inst)
            print(prediction)
            print(sample_data['action'])

            prediction_x.append(prediction[0])
            prediction_y.append(prediction[1])
            gt_x.append(sample_data['action'][0])
            gt_y.append(sample_data['action'][1])
    
    x_l = np.arange(20)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(x_l, prediction_x)
    axs[0].plot(x_l, gt_x)
    axs[0].legend(['prediction_x', 'ground_truth_x'])

    axs[1].plot(x_l, prediction_y)
    axs[1].plot(x_l, gt_y)
    axs[1].legend(['prediction_y', 'ground_truth_y'])

    plt.savefig('LTSim_inference_16.jpg')





    
    '''
    for step in steps:
        
        resized_im = resize_to_resolution(
            step['observation']['image'],
            to_numpy=False,
            target_width=256,
            target_height=256,
        )
        im = resized_im.numpy()
        
        #im = step['observation']['image'].numpy()
        inst = step['language_instruction'].numpy().decode('utf-8')
        #inst = step['observation']['language_instruction'].numpy().decode('utf-8')
        
        print(policy.generate_action(im, inst))
    '''