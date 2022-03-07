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
    model_re18 = load_model(model_dir+'resnet18_f1_normal', num_classes, device, 'MyMlpModelResnet18').to(device)
    model_re18.eval()
    model_eff = load_model(model_dir+'effib1_focal_default', num_classes, device, 'MyMlpModelEffiB1').to(device)
    model_eff.eval()
    model_vgg = load_model(model_dir+'vgg16_cross_mixup', num_classes, device, 'MyMlpModelVGG16').to(device)
    model_vgg.eval()

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

    preds = []

    m = torch.nn.Softmax(dim=1)

    minority_rank = [17, 11, 8, 14, 16, 10, 15, 9, 13, 12, 7, 6, 5, 2, 4, 3, 1, 0]
    
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = m(model_re18(images)) + m(model_eff(images)) + m(model_vgg(images))
            """ pred1 = (model_re18(images) + model_eff(images) + model_vgg(images))/2
            # print(pred.shape)
            images = torch.flip(images, dims=(-1,))
            pred2 = (model_re18(images) + model_eff(images) + model_vgg(images))/2

            pred = pred1 + pred2 """
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    """ mean_pred = np.array([])
    for i in range(2):
        if i == 0:
            mean_pred = preds[i]
        else:
            mean_pred += preds[i]
    
    final_pred = []

    for pred in mean_pred:
        final_pred.extend(pred.argmax(dim=-1)) """

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output_ensemble.csv')
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
    parser.add_argument('--augmentation', type=str, default='JuneCustomAug4', help='data augmentation type (default: BaseAugmentation)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
