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

def apply_over_channels(img, func):
    return np.stack([func(img[..., i]) for i in range(3)], 2)

class Image_Data(Dataset):
    def __init__(self, path, data_len, size, fourier_seed):
        self.path = path
        self.encrypt_partial = partial(encrypt, seed=self.fourier_seed)
        self.size = size
        fnames = os.listdir(path)
        random.seed(42)
        self.file_pairs = random.sample(list(combinations(fnames, 2)), data_len)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        file_pair= self.file_pairs[idx]
        x = cv2.imread(self.path + '/' + file_pair[0])
        x = cv2.resize(x, self.size)
        y = apply_over_channels(img, self.encrypt_partial)
        x, y = torch.tensor(x).transpose((2,0,1), torch.tensor(y).transpose((2,0,1)
        X = torch.stack((x.float(), y.float()))/255
        return X, X
    
def ImageLoader(path, data_len, fourierSeed = 42, size=300, bs=64):
    return DataLoader(Image_Data(path, data_len, fourierSeed, size), bs)
                                                                            
