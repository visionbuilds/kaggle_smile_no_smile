# import the necessary packages
import glob
import os
import matplotlib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = list(glob.iglob(root_dir + "/*.jpg"))
    def __len__(self):
        return len(self.all_images)
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image