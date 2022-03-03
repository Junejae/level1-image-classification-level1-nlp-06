import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import os, sys
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.autograd import Variable

from dataset import *
from model import *

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--device', type=str, default='cuda')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '../model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()


    data_dir = args.data_dir
    model_dir = args.model_dir

    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    print(args.device)
    device = args.device
    
    num_classes = MaskBaseDataset.num_classes  # 18
    model_path = os.path.join(args.model_dir, args.model_name, 'best.pth')
    print(model_path)
    model = MyEnsemble()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)
    
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = torch.argmax(pred, dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            
            
    info['ans'] = all_predictions
    
    info.to_csv("info.csv")

