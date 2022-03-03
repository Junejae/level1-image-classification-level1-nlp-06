import pandas as pd
import numpy as np
import random
import argparse
import os
from PIL import Image
import wandb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold


import json
import matplotlib.pyplot as plt
from model import *
from utils import *
from dataset import *

#import gc
#gc.collect()
#torch.cuda.empty_cache()


import warnings
warnings.filterwarnings("ignore")

SEED = 42


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def valid(val_loader, args, logger, best_val_acc, best_val_loss, val_set):
    # val loop
    with torch.no_grad():
        print("Calculating validation results...")
        model.eval()
        val_loss_items = []
        val_acc_items = []
        figure = None
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            loss_item = criterion(outs, labels).item()
            acc_item = (labels == preds).sum().item()
            val_loss_items.append(loss_item)
            val_acc_items.append(acc_item)

            if figure is None:
                inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                inputs_np = dataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                figure = grid_image(
                    inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                )

        val_loss = np.sum(val_loss_items) / len(val_loader)
        val_acc = np.sum(val_acc_items) / len(val_set)
        best_val_loss = min(best_val_loss, val_loss)
        if val_acc > best_val_acc:
            print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
            torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
            best_val_acc = val_acc
        torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
        print(
            f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
            f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
        )
        logger.add_scalar("Val/loss", val_loss, epoch)
        logger.add_scalar("Val/accuracy", val_acc, epoch)
        logger.add_figure("results", figure, epoch)
        
        wandb.log({
            "Val Accuracy": val_acc,
            "Val Loss": val_loss,
            "results": figure})
    
    return best_val_acc, best_val_loss
        
def train(train_loader, args, logger, epoch):
    # train loop
    model.train()
    loss_value = 0
    matches = 0

    for idx, train_batch in enumerate(train_loader):
        inputs, labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss = criterion(outs, labels)

        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        matches += (preds == labels).sum().item()
        if (idx + 1) % args.log_interval == 0:
            train_loss = loss_value / args.log_interval
            train_acc = matches / args.batch_size / args.log_interval
            current_lr = get_lr(optimizer)
            print(
                f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
            )
            logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
            logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
            
            wandb.log({
                "Train Accuracy": train_acc,
                "Train Loss": train_loss})
                                    



if __name__ == "__main__":
    seed_everything(42)

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_dir', type=str, default="/opt/ml/code/level1-image-classification-level1-nlp-6/special_mission/model")
    parser.add_argument('--data_dir', type=str, default="/opt/ml/input/data/train/images")
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--class_num', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--initial_lr', type=float, default=0.00001)
    parser.add_argument("--resize", nargs="+", type=list, default=[384, 288])
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--cross_valid', type=bool, default=False)
    args = parser.parse_args()
    
    wandb.init(project="hyunjin-project", entity="boostcamp-nlp06", name="resnet50+xavier+stepLR", config={
    "learning_rate": args.initial_lr,
    "epoch": args.epochs,
    "batch_size": args.batch_size,
    "resize": args.resize
    })
    
    config = wandb.config


    model_dir = args.model_dir
    data_dir = args.data_dir
    
    save_dir = increment_path(os.path.join(args.model_dir, args.model_name))
    device = args.device

    """
    Data Load & Preprocess
    """
    dataset = MaskSplitByProfileDataset(data_dir)
    transform = CustomAugmentation(resize=args.resize,
                                          mean=dataset.mean,
                                          std=dataset.std,)
    
    dataset.set_transform(transform)
    
    
    # -- model
    model = MyEnsemble()
    model = model.to(args.device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
    wandb.watch(model)
    
    criterion = LabelSmoothingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
    best_val_acc = 0
    best_val_loss = np.inf
    
    if args.cross_valid == True:
        kfold=KFold(n_splits=args.folds,shuffle=True)
        
        for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
            print('------------fold no.{}----------------------'.format(fold))
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            train_loader = torch.utils.data.DataLoader(
                                dataset, 
                                batch_size=args.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=args.batch_size, sampler=test_subsampler)


            for epoch in range(args.epochs):
                train(train_loader, args, logger, epoch)
                best_val_acc, best_val_loss = valid(val_loader, args, logger, best_val_acc, best_val_loss, test_idx)
    else:
        train_set, val_set = dataset.split_dataset()
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=True)
        for epoch in range(args.epochs):
            train(train_loader, args, logger, epoch)
            best_val_acc, best_val_loss = valid(val_loader, args, logger, best_val_acc, best_val_loss, val_set)
    
        
    print('done!')
    
