import os
import json
import copy
import torch
import transformers
import torchvision.transforms as T
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import Dataset

from src.train.arguments import DataArguments
from src.utils.constants import IGNORE_INDEX
from src.datasets.preprocess import preprocess, preprocess_multimodal


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 vision_tower: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 local_rank: int):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        if local_rank==0:
            print("Formatting inputs...Skip in lazy mode")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.vision_tower = vision_tower
        
        if self.data_args.image_aspect_ratio=="pad" and self.data_args.image_processor is None:
            from transformers import CLIPImageProcessor
            self.clip_processor = CLIPImageProcessor.from_pretrained("outputs/llava_ckpts/clip-vit-large-patch14-336")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        import pdb;pdb.set_trace()
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
  
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            
            if isinstance(self.data_args.image_folder, list):
                image_folder = self.data_args.image_folder[self.list_data_dict[i]['split']]
            else:
                image_folder = self.data_args.image_folder
            
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            processor = self.data_args.image_processor if hasattr(self.data_args, 'image_processor') else None
            
            if processor is None:
                image = image.resize((self.data_args.image_scale, self.data_args.image_scale))
        
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                pad_processor = processor if processor else self.clip_processor
                image = expand2square(image, tuple(int(x*255) for x in pad_processor.image_mean))
        
            if processor is None and "vae" in self.vision_tower:
                image = T.ToTensor()(image)
                if self.data_args.image_norm:
                    image = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
            else:
                #image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, do_resize=False)['pixel_values'][0]  # 3,224,224
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.data_args.image_processor is None and ("vae" in self.vision_tower or "unicode" in self.vision_tower):
                data_dict['image'] = torch.zeros(3, self.data_args.image_scale, self.data_args.image_scale)
            else:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, 
                                vision_tower,
                                local_rank=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                vision_tower=vision_tower,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                local_rank=None)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
