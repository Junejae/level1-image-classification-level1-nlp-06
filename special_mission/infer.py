import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import os, sys
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.autograd import Variable

from Dataset import CustomDataset

WORKERS = 2
CHANNEL = 3

import warnings
warnings.filterwarnings("ignore")
IMG_SIZE = 512
NUM_CLASSES = 18
SEED = 42
device = "cuda"

torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED) 

transform = transforms.Compose([Resize((512, 384), Image.BILINEAR),
                                ToTensor(),
                                Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))])


if __name__ == "__main__":
    test_df = pd.read_csv("../../../input/data/eval/info.csv")
    
    
    test_dataset = CustomDataset(test_df, transform = transform, train=False)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=8,
                                                    shuffle=False)
    
    model = torch.load("../model/baseline_model.pt")
    model.eval()
    
    all_predictions = []
    for images in test_dataloader:
        with torch.no_grad():
            images = Variable(images).to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            
            
    test_df['ans'] = all_predictions
    
    test_df.to_csv("info.csv")

