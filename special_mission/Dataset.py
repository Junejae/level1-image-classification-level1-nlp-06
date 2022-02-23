# 출처: https://github.com/utkuozbulak/pytorch-custom-dataset-examples/blob/master/src/custom_dataset_from_file.py
import numpy as np
from PIL import Image
import glob
from torchvision import transforms
import os
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import Resize, ToTensor, Normalize




class CustomDataset(Dataset):
    def __init__(self, df_train, transform, train=True):
        self.transform = transform
        self.train = train
        
        if train:
            self.image_list = df_train['path'].tolist()
            self.target = df_train['class'].tolist()
        else:
            self.image_list = ["../../../input/data/eval/images/"+f for f in df_train['ImageID'].tolist() if "._" not in f]

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        image = Image.open(single_image_path)

        if self.transform:
            img = self.transform(image)
    
        if self.train:
            label = self.target[index]
            
            return (img, torch.tensor(label))
        else:
            return img

    def __len__(self):
        return len(self.image_list)