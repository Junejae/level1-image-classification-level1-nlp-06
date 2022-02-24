from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

class TrainDataset(Dataset):
    # input: image_list, target_list
    def __init__(self, X, y, transform=None):
        self.image_paths = X
        self.target = y
        self.transform = transform
    
    # output: PIL_image, label
    def __getitem__(self, index):
        images = []
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        
        return (image, self.target[index])
    
    def __len__(self):
        return len(self.image_paths)