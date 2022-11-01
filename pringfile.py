import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation
import pdb
import transforms as T
import utils
import numpy as np
import shutil, glob

from args import get_parser

parser = get_parser()
args = parser.parse_args()
log_dir = os.path.join(args.rootpath, args.model_id)
# copy_dirs(log_dir)
# shutil.copytree('./lib/', os.path.join(work_dir, 'lib'))
# os.system('cp %s %s' % ('./lib/', log_dir))
LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)