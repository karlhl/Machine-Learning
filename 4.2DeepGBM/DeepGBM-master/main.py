import argparse, os, logging, random, time
import numpy as np
import math
import time
import scipy.sparse
import lightgbm as lgb
import torch
import torch.nn as nn
# import torchvision
from torch.autograd import Variable

from train_models import *

parser = argparse.ArgumentParser(description = 'DeepGBM Models')
parser.add_argument('-data', type = str, default = 'YAHOO')
parser.add_argument('-model', type = str, default = 'deepgbm')

parser.add_argument('-batch_size', type = int, default = 128)
parser.add_argument('-test_batch_size', type = int, default = 50000)

parser.add_argument('-seed', type = str, default = '1')# '1,2,3,4,5'
parser.add_argument('-log_freq', type = int, default = 100)
parser.add_argument('-test_freq', type = int, default = 1000)

parser.add_argument('-l2_reg', type = float, default = 1e-6)
parser.add_argument('-l2_reg_opt', type = float, default = 5e-4)
parser.add_argument('-plot_title', type = str, default = None)

parser.add_argument('-emb_epoch', type = int, default = 1)
parser.add_argument('-emb_lr', type = float, default = 1e-3)
parser.add_argument('-emb_opt', type = str, default = "Adam")

parser.add_argument('-nslices', type = int, default = 10)
parser.add_argument('-ntrees', type = int, default = 100)

parser.add_argument('-tree_layers', type = str, default = "10,5")
parser.add_argument('-cate_layers', type = str, default = "32,32")

parser.add_argument('-maxleaf', type = int, default = 128)
parser.add_argument('-mindata', type = int, default = 40)
parser.add_argument('-tree_lr', type = float, default = 0.15)
parser.add_argument('-embsize', type = int, default = 20)

parser.add_argument('-lr', type = float, default = 1e-3)
parser.add_argument('-opt', type = str, default = 'AdamW')

parser.add_argument('-max_epoch', type = int, default = 50)
parser.add_argument('-loss_init', type = float, default = 1.0)
parser.add_argument('-loss_dr', type = float, default = 0.9)

parser.add_argument('-group_method', type = str, default = 'Random')
parser.add_argument('-feature_emb_size', type = int, default = 50)

parser.add_argument('-feat_per_group', type = int, default = 128)
parser.add_argument('-loss_de', type = int, default = 5)
parser.add_argument('-task', type = str, default = 'regression')


args = parser.parse_args()
assert(args.nslices <= args.ntrees)

plot_title = args.plot_title

args.seeds = [int(x) for x in args.seed.split(',')]
random.seed(args.seeds[0])
np.random.seed(args.seeds[0])
torch.cuda.manual_seed_all(args.seeds[0])
    
def main():
    if "gbdt2nn" in args.model:
        num_data = dh.load_data(args.data+'_num')
        train_GBDT2NN(args, num_data, plot_title)
    elif args.model == "deepgbm":
        num_data = dh.load_data(args.data+'_num')
        cate_data = dh.load_data(args.data+'_cate')
        # designed for faster cateNN
        cate_data = dh.trans_cate_data(cate_data)
        train_DEEPGBM(args, num_data, cate_data, plot_title)
    
if __name__ == '__main__':
    main()
