import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

import numpy as np


def load_model(saved_model, num_classes, device, model_name):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model_age = load_model(model_dir+'age22', 3, device, 'MyFcModel').to(device)
    model_age.eval()
    model_gender = load_model(model_dir+'gender', 2, device, 'MyFcModel').to(device)
    model_gender.eval()
    model_mask = load_model(model_dir+'mask', 3, device, 'MyFcModel').to(device)
    model_mask.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)

    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")

    preds1, preds2, preds3 = [], [], []

    m = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred1 = m(model_age(images)) 
            pred2 = m(model_gender(images)) 
            pred3 = m(model_mask(images))

            pred1 = pred1.argmax(dim=-1)
            pred2 = pred2.argmax(dim=-1)
            pred3 = pred3.argmax(dim=-1)

            preds1.extend(pred1.cpu().numpy())
            preds2.extend(pred2.cpu().numpy())
            preds3.extend(pred3.cpu().numpy())

    preds = []

    for i in range(len(preds1)):
        x, y, z = preds1[i], preds2[i], preds3[i]

        preds.append(x + y*3 + z*6)


    info['ans'] = preds
    save_path = os.path.join(output_dir, f'trinity2.csv')
    info.to_csv(save_path, index=False)
    print(info.ans.value_counts())
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=[512, 384], help='resize size for image when you trained (default: (96, 128))')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # Custom arguments  
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
