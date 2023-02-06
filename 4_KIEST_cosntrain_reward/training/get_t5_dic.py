#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os

import torch
import sys
import numpy as np
from tqdm import tqdm
import random

# to avoid "src.xxx" not found error.
sys.path.insert(0, '..')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


fw=open("t5_dic.txt","w")
tokenizer = AutoTokenizer.from_pretrained('t5-large')  # Fixed GPT2 tokenizer.
s=tokenizer.get_vocab()
for word in s:
    fw.write(word+"\n")
