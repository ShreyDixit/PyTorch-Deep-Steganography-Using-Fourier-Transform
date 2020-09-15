# encoding: utf-8
"""
@author: Shrey Dixit
@contact: shreyakshaj@gmail.com
@version: 1.0
@file: main.py
@time: 2020/09/14
"""

import os
import torch
import argparse
import numpy
from functools import partial

from utils import *
from model import StegNet

from fastai.core import parallel
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.train import fit_one_cycle, lr_find

parser = argparse.ArgumentParser()

group1 = parser.add_mutually_exclusive_group()
group1.add_argument('-use', action='store_true',  help='Use for Inference')
group1.add_argument('-train', action='store_true',  help='Train a new model')

group2 = parser.add_mutually_exclusive_group()
group2.add_argument('-encode', action='store_true',  help='Use encoder to encode secret image in cover image')
group2.add_argument('-decode', action='store_true',  help='Use decoder to decode secret image from cover image')

parser.add_argument('--datapath', type=str, metavar='', default='data',  help='Path to Dataset folder')
parser.add_argument('--num_train', type=str, metavar='', default=50000,  help='Number of training pairs to be created')
parser.add_argument('--num_val', type=str, metavar='', default=1000,  help='Number of validation pairs to be created')
parser.add_argument('--size', metavar='', default=300, help='Size of the images')
parser.add_argument('--bs', metavar='', default=64, help='Batch Size')
parser.add_argument('--epochs', metavar='', default=10, help='Number of Epochs')
parser.add_argument('--model', metavar='', default=None, help='Path for the model file if you want to finetune')
parser.add_argument('--fourierSeed', metavar='', default=42, 
                    help='Seed for generating the pseudorandom matrix for changing phase in fourier domain')

args = parser.parse_args()

def main():
    model = StegNet(8, 4)
    
    if args.train:
        data_train = DataLoader(args.datapath + '/train', args.num_train, args.fourierSeed, args.size, args.bs)
        data_train = DataLoader(args.datapath + '/val', args.num_val, args.fourierSeed, args.size, args.bs)
        data = DataBunch(data_train, data_val)
        
        if args.model is not None:
            model.load_state_dict(torch.load(args.model))
            
        loss_fn = mse
        
        learn = Learner(data, model, loss_func = loss_fn, metrics = [mse_cov, mse_hidden])
        fit_one_cycle(learn, args.epochs, 3e-2)
        
        torch.save(learn.model.state_dict())

    else:
        path = input("Enter path of the model: ") if args.model is None else args.model
        model.load_state_dict(torch.load(args.model))
        model.eval()
        
        if args.encode:
            f_paths = [args.datapath + '/'+ f for f in os.listdir(args.datapath + '/cover')]
            try:  
                os.mkdir(args.datapath+'/encoded')
            except OSError:
                pass
            fourier_func = partial(encrypt, seed = args.seed)
            encode_partial = partial(encode, model=model.encoder, size=size, fourier_func)
            parallel(encode_partial, f_paths)
        
        else: 
            f_paths = [args.datapath + '/'+ f for f in os.listdir(args.datapath + '/encoded')]
            try:  
                os.mkdir(args.datapath+'/decoded')
            except OSError:
                pass
            fourier_func = partial(decrypt, seed = args.seed)
            decode_partial = partial(decode, model=model.decoder, size=size, fourier_func)
            parallel(decode_partial, f_paths)
            
if __name__ == '__main__':
    main()
