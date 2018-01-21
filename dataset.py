from torch.utils.data import Dataset
import pandas as pd
from glob import glob
import os
import torch
import re
from PIL import Image

class ImagesDataset(Dataset):
    """Images dataset."""

    def __init__(self, root_dir, transform=None, number_sort=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            number_sort (bool, optional): If True, sort files by numerical order. If False, sort files by ascii order.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        def numericalSort(value):
          numbers = re.compile(r'(\d+)')
          parts = numbers.split(value)
          parts[1::2] = map(int, parts[1::2])
          return parts
        
        if number_sort:
          self.file_list = sorted(glob(os.path.join(root_dir, "*.png")), key=numericalSort)
        else:
          self.file_list = sorted(glob(os.path.join(root_dir, "*.png")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample