import datetime
import logging
import logging.handlers
import os
import sys
from transformers import AutoConfig
import random
import pickle
import requests
import numpy as np
import torch
import torch.distributed as dist
from src.utils.constants import LOGDIR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def is_master_proc():
    return dist.get_rank()==0


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def compute_p_norm(model):
    norm = 0
    for k, v in model.state_dict().items():
        v = v.detach().clone()
        norm += torch.sum(v.view(-1).pow_(2))
    return norm


def get_num_conv_linear_layers(model):
    cnt = 0
    weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if pn.endswith('weight') and isinstance(m, weight_modules):
                cnt += 1
    return cnt


def compute_model_size(model, logger):
    if logger is not None:
        logger.info(
            "#parameters: %.4fM", sum(p.numel() for p in model.parameters()) / 1000 / 1000
        )


def set_seed(seed=None):
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def np2tn(array):
    if len(array.shape) == 4:
        return torch.from_numpy(np.transpose(array, (3, 2, 0, 1)))
    elif len(array.shape) == 2:
        return torch.from_numpy(array.T)
    else:
        raise ValueError('invalid shape')


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)

    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def top_p_probs(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return norm_probs


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Take a 2-dim tensor, apply softmax along each row, and sample from
    each multinomial distribution defined by the rows.

    Args:
        logits: 2-dim tensor of shape (n_samples, logit_dim)
        temperature (float): softmax temperature
        top_k (Optional[int]): if given, sample only using `top_k` logits
        top_p (Optional[float]): if given, sample only using `top_p` logits

    Returns:
        samples: 1-dim integer tensor of shape (n_samples,)
    """

    logits = logits.to(dtype=torch.float32)
    logits = logits / temperature
    
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')

    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    try:
        samples = torch.multinomial(probs, num_samples=1)
    except RuntimeError:
        print(probs)
        print(logits)
        print('isinf, ', torch.sum(torch.isinf(probs)))
        print('isnan, ', torch.sum(torch.isnan(probs)))
        print('is negative', torch.sum(probs < 0))
        raise

    return samples.view(-1)


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t
