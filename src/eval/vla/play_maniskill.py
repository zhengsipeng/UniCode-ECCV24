import numpy as np
import torch
import argparse

from src.eval.vla.action_inference import VLAPolicy
from data.ManiSkill2.eval_policy_multi import eval_lvm

if __name__ == "__main__":
    default_model_path = 'outputs/vla_ckpts/vla_vicuna_lora_mmload_taco_0227_4layers_2epoch'
    default_model_base = 'outputs/llava_ckpts/vicuna-7b-v1.5'
    default_output_layer = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=default_model_path)
    parser.add_argument("--model-base", type=str, default=default_model_base)
    parser.add_argument("--instruction", type=str, default="Pick up the object and move it to a goal position.")
    parser.add_argument("--output-layer", type=int, default=default_output_layer)
    parser.add_argument("--record-dir", type=str)
    args = parser.parse_args()

    policy = VLAPolicy(args.model_path, args.model_base, output_layer=args.output_layer, embodiment='maniskill')

    eval_lvm(model=policy, instruction=args.instruction, record_dir=args.record_dir)

    
