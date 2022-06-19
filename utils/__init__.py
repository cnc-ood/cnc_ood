import logging
from .logger import *
from .eval import *
from .argparser import parse_args
from .misc import *
import torch
import os
import shutil

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f"model_best.pth"))

def create_save_path(args):
    ans_str = f"_{args.model}_{args.aug}"
    return ans_str

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False