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

from data import ImageLoader
from model import StegNet

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

def decrypt(img_channel, seed = 42):
    np.random.seed(seed)
    f = np.fft.fft2(img_channel)
    mag, ang = np.abs(f), np.arctan2(f.imag, f.real)
    ns = np.random.uniform(0, 6.28, size = f.shape)
    ang_rec = ang-ns
    img_rec = np.fft.ifft2(mag*np.exp((ang-ns)*1j)).real
    return img_rec

def mse(y_pred, y): return (((y_pred - y)*255)**2).mean()
def mse_cov(y_pred, y): return (((y_pred[0] - y[0])*255)**2).mean()
def mse_hidden(y_pred, y): return (((y_pred[1] - y[1])*255)**2).mean()

def main():
    if args.train:
        data_train = DataLoader(args.datapath + '/train', args.num_train, args.fourierSeed, args.size, args.bs)
        data_train = DataLoader(args.datapath + '/val', args.num_val, args.fourierSeed, args.size, args.bs)
        data = DataBunch(data_train, data_val)
        
        model = StegNet(8, 4)
        if args.model is not None:
            model.load_state_dict(torch.load(args.model))
            
        loss_fn = mse
        
        learn = Learner(data, model, loss_func = loss_fn, metrics = [mse_cov, mse_hidden])
        fit_one_cycle(learn, args.epochs, 3e-2)
        
        torch.save(learn.model.state_dict())
