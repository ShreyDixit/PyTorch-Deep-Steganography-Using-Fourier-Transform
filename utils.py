import numpy as np
import os
import torch
import cv2
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import torchvision.transforms as transforms

def encrypt(img_channel, seed = 42):
    np.random.seed(seed)
    f = np.fft.fft2(img_channel)
    mag, ang = np.abs(f), np.arctan2(f.imag, f.real)
    ns = np.random.uniform(0, 6.28, size = f.shape)
    ang_new = ang+ns
    noise_img = np.fft.ifft2(mag*np.exp((ang_new)*1j)).real
    return noise_img

def decrypt(img_channel, seed = 42):
    np.random.seed(seed)
    f = np.fft.fft2(img_channel)
    mag, ang = np.abs(f), np.arctan2(f.imag, f.real)
    ns = np.random.uniform(0, 6.28, size = f.shape)
    ang_new = ang-ns
    noise_img = np.fft.ifft2(mag*np.exp((ang_new)*1j)).real
    return noise_img

def image_crypto(f_path, size, func):
    img = cv2.imread(f_path)[:, :, [2, 1, 0]]
    img = cv2.resize(img)
    return np.stack([func(img[..., i]) for i in range(3)], 2)

def encode(f_path, idx, model, size, fourier_func):
    path, fname = f_path.split('/')[:-2].join('/'), f_path.split('/')[-1]
    x = cv2.resize(cv2.imread(f_path)[:, :, [2, 1, 0]], size)
    y = image_crypto(path + '/secret/'+ fname, size, fourier_func)
    x, y = torch.tensor(x).transpose((2,0,1), torch.tensor(y).transpose((2,0,1)
    X = torch.stack((x.float(), y.float()))/255
    out = model(X[None]).squeeze().permute(1, 2, 0)[:, :, [2, 1, 0]]
    cv2.imwrite(path + '/encoded/' + fname, out)
                                                                        
def encode(f_path, idx, model, size, fourier_func):
    path, fname = f_path.split('/')[:-2].join('/'), f_path.split('/')[-1]
    x = cv2.resize(cv2.imread(f_path)[:, :, [2, 1, 0]], size)
    x = torch.tensor(x).transpose((2,0,1)
    out = model(x[None]).squeeze().permute(1, 2, 0)[:, :, [2, 1, 0]]
    cv2.imwrite(path + '/decoded/' + fname, out)                                                                        
                                                                            
def mse(y_pred, y): return (((y_pred - y)*255)**2).mean()
def mse_cov(y_pred, y): return (((y_pred[0] - y[0])*255)**2).mean()
def mse_hidden(y_pred, y): return (((y_pred[1] - y[1])*255)**2).mean()                                                                      

class Image_Data(Dataset):
    def __init__(self, path, data_len, size, fourier_seed):
        self.path = path
        self.encrypt_partial = partial(encrypt, seed=self.fourier_seed)
        self.size = size
        f_paths = [path + '/'+ f for f in os.listdir(path)]
        random.seed(42)
        self.file_pairs = random.sample(list(combinations(f_paths, 2)), data_len)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        file_pair= self.file_pairs[idx]
        x = cv2.imread(self.path + '/' + file_pair[0])[:, :, [2, 1, 0]]
        x = cv2.resize(x, self.size)
        y = image_crypto(self.path + '/' + file_pair[1], self.size, self.encrypt_partial)
        x, y = torch.tensor(x).transpose((2,0,1), torch.tensor(y).transpose((2,0,1)
        X = torch.stack((x.float(), y.float()))/255
        return X, X
    
def ImageLoader(path, data_len, fourierSeed = 42, size=300, bs=64):
    return DataLoader(Image_Data(path, data_len, fourierSeed, size), bs)
                                                                            