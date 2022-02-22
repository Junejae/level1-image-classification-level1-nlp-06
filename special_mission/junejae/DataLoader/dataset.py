from torch.utils.data.dataset import dataset
from torchvision import transforms
import pandas as pd
import numpy as np

class MyCustomDataset(Dataset):
    def __init__(self, csv_path):

        self.to_tensor = transforms.ToTensor()

        self.data_info = pd.read_csv(csv_path, header=None)

        self.image_arr = np.asarray(self.data_info.iloc[:, 0])

        self.label_arr = 
    
    def __getitem__(self, index):


        return (img, label)

    def __len__(self):

        return count

if __name__ == '__main__':
