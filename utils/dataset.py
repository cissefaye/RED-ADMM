from glob import glob
import cv2
import os
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import numpy as np


class GetDataset(Dataset):
    def __init__(self, root: str, img_size: (int, int) =(256, 256)):

        self.fpaths = sorted(glob(root + '/*.png', recursive=True))
        self.transforms = transforms.Compose([
            transforms.ToTensor(), 
        ])
        self.img_size = img_size
        
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = cv2.imread(fpath)
        img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.transforms(img)