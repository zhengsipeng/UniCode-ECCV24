import torch
import transformers
import numpy as np
from .constants import DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN
from PIL import Image

WV_MIN = -2.0
WV_MAX = 2.0
RD_MIN = -1.57
RD_MAX = 1.57
NUM_BINS = 256
ACTION_TOKEN_BIAS = 3
ACTION_DIM = 7
#TD_BINS = 21
TD_BINS = 41
TD_DIM = 2

def action_tokenizer(action_vector: np.ndarray, tokenizer, embodiment):
    # Language Table: 2d action
    if len(action_vector) == 2:
        if embodiment == 'language_table':
            ranges_min = [-0.25, -0.25]
            ranges_max = [0.25, 0.25]
        else:
            ranges_min = [-0.03, -0.03]
            ranges_max = [0.03, 0.03]

        discretized = np.zeros_like(action_vector, dtype=int)
        for i in range(2):
            discretized[i] = np.floor((action_vector[i] - ranges_min[i]) / (ranges_max[i] - ranges_min[i]) * TD_BINS).astype(int)
            if discretized[i] >= TD_BINS:
                discretized[i] = TD_BINS - 1
            if discretized[i] < 0:
                discretized[i] = 0

        action_tokens = list(tokenizer.get_vocab().keys())[ACTION_TOKEN_BIAS:TD_BINS+ACTION_TOKEN_BIAS]

        input_tokens = [str(action_tokens[i]) for i in discretized]
        action_str = " ".join(input_tokens)
        action_str = DEFAULT_ACT_START_TOKEN + action_str + DEFAULT_ACT_END_TOKEN

        return action_str


    # into integers
    ranges_min = [WV_MIN, WV_MIN, WV_MIN, RD_MIN, RD_MIN, RD_MIN, -1.0]
    ranges_max = [WV_MAX, WV_MAX, WV_MAX, RD_MAX, RD_MAX, RD_MAX, 1.0]
    
    discretized = np.zeros_like(action_vector, dtype=int)
    
    for i in range(ACTION_DIM):
        discretized[i] = np.floor((action_vector[i] - ranges_min[i]) / (ranges_max[i] - ranges_min[i]) * NUM_BINS).astype(int)
        if discretized[i] >= NUM_BINS:
            discretized[i] = NUM_BINS - 1
        if discretized[i] < 0:
            discretized[i] = 0
    # into tokens
    action_tokens = list(tokenizer.get_vocab().keys())[ACTION_TOKEN_BIAS:NUM_BINS+ACTION_TOKEN_BIAS]
    #action_token_ids = tokenizer.convert_tokens_to_ids(action_tokens)
    
    #input_ids = [action_token_ids[i] for i in discretized]
    input_tokens = [str(action_tokens[i]) for i in discretized]
    action_str = " ".join(input_tokens)
    action_str = DEFAULT_ACT_START_TOKEN + action_str + DEFAULT_ACT_END_TOKEN
    
    return action_str

def action_denormalize(normalized_action, embodiment):
    env_action = normalized_action
    if embodiment == 'maniskill':
        for i in range(3):
            env_action[i] = env_action[i] / 20.0
        for i in range(3,6):
            env_action[i] = env_action[i] / 15.7
        return env_action
    else:
        return env_action

def decode_action(output_ids, tokenizer, embodiment):
    if embodiment == 'language_table' or embodiment == 'language_table_sim':
        target_dim = TD_DIM
        target_bins = TD_BINS
    else:
        target_dim = ACTION_DIM
        target_bins = NUM_BINS

    dummy_action = np.zeros(target_dim)
    if target_dim == 7:
        dummy_action[-1] = -1.0
    
    if len(output_ids) != target_dim:
        print(f"Invalid action dimension: {len(output_ids)}.")
        return action_denormalize(dummy_action, embodiment)
    
    action_tokens = list(tokenizer.get_vocab().keys())[ACTION_TOKEN_BIAS:target_bins+ACTION_TOKEN_BIAS]
    action_token_ids = tokenizer.convert_tokens_to_ids(action_tokens)
    
    discretized = []
    for ids in output_ids:
        if ids not in action_token_ids:
            print("Invalid action token used.")
            return action_denormalize(dummy_action, embodiment)
        discretized.append(action_token_ids.index(ids))
    
    action_vector = np.empty_like(discretized, dtype=float)
    if embodiment == 'language_table':
        ranges_min = [-0.25, -0.25]
        ranges_max = [0.25, 0.25]
    elif embodiment == 'language_table_sim':
        ranges_min = [-0.03, -0.03]
        ranges_max = [0.03, 0.03]
    else:
        ranges_min = [WV_MIN, WV_MIN, WV_MIN, RD_MIN, RD_MIN, RD_MIN, -1.0]
        ranges_max = [WV_MAX, WV_MAX, WV_MAX, RD_MAX, RD_MAX, RD_MAX, 1.0]
    
    for i in range(target_dim):
        action_vector[i] = ((float(discretized[i]) + 0.5) / target_bins) * (ranges_max[i] - ranges_min[i]) + ranges_min[i]
    
    # gripper_dim should be -1.0 or 1.0
    if target_dim == 7:
        if action_vector[-1] > 0.0:
            action_vector[-1] = 1.0
        else:
            action_vector[-1] = -1.0
    
    return action_denormalize(action_vector, embodiment)
    

def concatenate_trajectory(image_1, image_2, dim=1):
    return np.concatenate([image_1, image_2], dim)
    
def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
        
    