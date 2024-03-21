import argparse

from src.eval.vla.action_inference import VLAPolicy
from Language_Table.eval_language_table import eval_lvm

if __name__ == "__main__":
    default_model_path = 'outputs/vla_ckpts/vla_vicuna_lora_mmload_languageT_0312_8layers_32batch'
    default_model_base = 'outputs/llava_ckpts/vicuna-7b-v1.5'
    default_output_layer = 8

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=default_model_path)
    parser.add_argument("--model-base", type=str, default=default_model_base)
    parser.add_argument("--output-layer", type=int, default=default_output_layer)
    parser.add_argument("--record-dir", type=str)
    args = parser.parse_args()

    policy = VLAPolicy(args.model_path, args.model_base, output_layer=args.output_layer, embodiment='language_table_sim')

    reward, length, success = eval_lvm(model=policy, record_dir=args.record_dir)

    print(reward, length, success)

    
