import numpy as np
import os
import torch
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import torchvision.transforms as transforms


transformations = transforms.Compose([
    transforms.ToTensor(),
])

class Image_Data(Dataset):
    def __init__(self, path, data_len):
        fnames = os.listdir(path)
        random.seed(42)
        self.file_pairs = random.sample(list(combinations(fnames, 2)), data_len)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        file_pair= self.file_pairs[idx]
        x = transformations(Image.open(f"ImageData/original/{file_pair[0]}"))
        y = torch.tensor(np.load(f"ImageData/fourier/{file_pair[1][:-3]}npy").transpose((2,0,1))/255)
        X = torch.stack((x,y.float()))
        return X, X
    
def ImageLoader(path, data_len, bs=64):
    return DataLoader(Image_Data(path, data_len), bs)
