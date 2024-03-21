from PIL import Image
from io import BytesIO
import re
import base64

import torch
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX, IMAGE_GEN_TOKEN_INDEX, DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    #import pdb;pdb.set_trace()
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_gen(prompt, tokenizer, sep, return_tensors=None):
    if len(prompt.split(sep))==2:
        query, answer = prompt.split(sep)
    else:
        query = prompt.split(sep)[0]
        answer = ""

    # query tokens
    pre_seq = query.split("<vistok>")[0]
    input_ids = tokenizer(pre_seq).input_ids

    vis_tokens_seq = re.findall(r"<vistok>(.*?)</vistok>", query)
    if len(vis_tokens_seq)>0:
        assert len(vis_tokens_seq)==1
        if vis_tokens_seq[0]!="":
            input_ids += [int(tok) for tok in vis_tokens_seq[0].split("-")] #[image_token_index] + visual_tokens + [image_token_index]

    # seq tokens
    input_ids += tokenizer(sep).input_ids[1:]
    
    # answer tokens
    if answer!="":
        vis_tokens_seq = re.findall(r"<vistok>(.*?)</vistok>", answer)
        input_ids += [int(tok) for tok in vis_tokens_seq[0].split("-")]
        post_seq = answer.split("</vistok>")[-1]
        input_ids += tokenizer(post_seq).input_ids[1:]
        
    #
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    
    return input_ids


def tokenizer_image_token_with_action(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None, use_act_tag=True):
    #prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    prompt_chunks = []
    for chunk in prompt.split('<image>'):
        if DEFAULT_ACT_START_TOKEN in chunk:
            # assert 1 start & 1 end
            if use_act_tag:
                before_act = chunk.split(DEFAULT_ACT_START_TOKEN)[0] + DEFAULT_ACT_START_TOKEN
            else:
                # omit the last space
                before_act = chunk.split(DEFAULT_ACT_START_TOKEN)[0][:-1]
            act_tokens = (chunk.split(DEFAULT_ACT_START_TOKEN)[1]).split(DEFAULT_ACT_END_TOKEN)[0]
            if use_act_tag:
                after_act = DEFAULT_ACT_END_TOKEN + chunk.split(DEFAULT_ACT_END_TOKEN)[1]
            else:
                after_act = chunk.split(DEFAULT_ACT_END_TOKEN)[1]

            #import pdb;pdb.set_trace()
            before_ids = tokenizer(before_act).input_ids
            after_ids = tokenizer(after_act).input_ids
            if after_ids[0] == tokenizer.bos_token_id:
                after_ids = after_ids[1:]
            act_ids = tokenizer.convert_tokens_to_ids(act_tokens.split())
            chunk_ids = before_ids + act_ids + after_ids
            prompt_chunks.append(chunk_ids)
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    #import pdb;pdb.set_trace()
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
