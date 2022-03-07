from preprocess import preprocessFunction
from torch.utils.data.dataset import dataset
from torchvision import transforms
import pandas as pd
import numpy as np


class MyCustomDataset(Dataset):
    def __init__(self, path):
        # assuming path = '/opt/ml/input/data/train'
        self.path = path
        self.data_info = preprocessFunction()

        self.to_tensor = transforms.ToTensor()

        self.image_arr = np.asarray(self.data_info.img_path)
        self.label_arr = np.asarray(self.data_info.label)

        self.data_len = len(self.data_info.id)

    def __getitem__(self, index):
        img_path = path + "/images/" + self.image_arr[index]
        img_thing = Image.open(img_path)

        img_tensor = self.to_tensor(img_thing)
        img_label = self.label_arr[index]

        return (img_tensor, img_label)

    def __len__(self):
        return self.data_len
