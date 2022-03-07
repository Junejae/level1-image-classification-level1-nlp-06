import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

# Custom
import wandb
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch import cuda
import pandas as pd
from glob import glob

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"] 
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
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


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# Functions for mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# end of  mixup functions

# Start of Custom split
def splitEvenlyAndOversampled(train_dir):
    prepro_data_info_train = pd.DataFrame(columns={'id','img_path','race','mask','gender','age','label'})
    prepro_data_info_valid = pd.DataFrame(columns={'id','img_path','race','mask','gender','age','label'})

    all_id_train, all_path_train, all_race_train, all_mask_train, all_age_train, all_gender_train, all_label_train = [],[],[],[],[],[],[]
    all_id_valid, all_path_valid, all_race_valid, all_mask_valid, all_age_valid, all_gender_valid, all_label_valid = [],[],[],[],[],[],[]

    current_class_capacity = [18, 18, 18, 18, 18, 18]
    valid_id_list = []
    
    for absolute_path in glob(train_dir + "/*/*"):

        split_list = absolute_path.split("/")
        img_name = split_list[-1]
        img_path = split_list[-2]

        path_split = img_path.split("_")

        img_id = path_split[0]
        img_gender = 0 if path_split[1] == "male" else 1
        img_race = path_split[2]
        img_age = min(2, int(path_split[3]) // 30)

        img_mask = 0
        if 'incorrect' in img_name:
            img_mask = 1
        elif 'normal' in img_name:
            img_mask = 2

        # -- 미스라벨링 교정 시작
        # -- Swap Gender
        if img_id in ['000225','000664','000767','001498-1','001509','003113','003223','004281','004432','005223','006359',
                '006360','006361','006362','006363','006364','006424','000667','000725','000736','000817','003780','006504']:
            temp = 0 if img_gender == 1 else 1
            img_gender = temp
        
        # -- Change Age to ~29
        if img_id in ['001009','001064','001637','001666','001852']:
            img_age = 0
        
        # -- Change Age to 60~
        if img_id in ['004348']: # 고민거리, 이분은 액면가는 폭삭 늙으셨지만 59세로 찍혀 있는데 과연 이걸 60대 노인 취급해도 될지 안 될지...
            img_age = 2

        # -- Correct Mask Status, normal <-> incorrect
        if img_id in ['000020','004418','005227']:
            if img_mask != 0:
                temp = 1 if img_mask == 2 else 2
                img_mask = temp
        # -- 미스라벨링 교정 끝

        
        # Check if it can go to the Valid Set
        is_train = True

        if img_id in valid_id_list:
            is_train = False
        
        elif current_class_capacity[img_gender*3 + img_age] != 0:
            current_class_capacity[img_gender*3 + img_age] -= 1
            valid_id_list.append(img_id)
            is_train = False


        if is_train:
            # oversampling
            n = 1
            if (img_age == 1 and img_gender == 0):
                n *= 2
            if img_age == 2:
                n *= 10
            if img_mask != 0:
                n *= 5
            for _ in range(n):
                all_id_train.append(img_id)
                all_path_train.append(absolute_path)
                all_race_train.append(img_race)
                all_mask_train.append(img_mask)
                all_gender_train.append(img_gender)
                all_age_train.append(img_age)
                all_label_train.append(img_mask*6 + img_gender*3 + img_age)
        else:
            # oversampling
            n = 1
            if img_mask != 0:
                n *= 4
            for _ in range(n):
                all_id_valid.append(img_id)
                all_path_valid.append(absolute_path)
                all_race_valid.append(img_race)
                all_mask_valid.append(img_mask)
                all_gender_valid.append(img_gender)
                all_age_valid.append(img_age)
                all_label_valid.append(img_mask*6 + img_gender*3 + img_age)
        

    prepro_data_info_train['id'] = all_id_train
    prepro_data_info_train['img_path'] = all_path_train
    prepro_data_info_train['race'] = all_race_train
    prepro_data_info_train['mask'] = all_mask_train
    prepro_data_info_train['gender'] = all_gender_train
    prepro_data_info_train['age'] = all_age_train
    prepro_data_info_train['label'] = all_label_train

    prepro_data_info_valid['id'] = all_id_valid
    prepro_data_info_valid['img_path'] = all_path_valid
    prepro_data_info_valid['race'] = all_race_valid
    prepro_data_info_valid['mask'] = all_mask_valid
    prepro_data_info_valid['gender'] = all_gender_valid
    prepro_data_info_valid['age'] = all_age_valid
    prepro_data_info_valid['label'] = all_label_valid
    
    return prepro_data_info_train, prepro_data_info_valid

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    train_list, val_list = splitEvenlyAndOversampled(data_dir)

    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset

    train_set = dataset_module(train_list)
    val_set = dataset_module(val_list)

    num_classes = train_set.num_classes  # 18
    

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=train_set.mean,
        std=train_set.std,
    )
    
    train_set.set_transform(transform)
    val_set.set_transform(transform)

    # -- data_loader

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.lr_decay_step, T_mult=1, eta_min=0.000001)

    # -- logging
    
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    

    # Setup WandB
    wandb.init(project="Junejae-Experiment", entity="boostcamp-nlp06", name="vgg16+cross+mixup")
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0 # for f1

    """ # Gradient Accumulation
    NUM_ACCUM = 2
    optimizer.zero_grad() """

    for epoch in range(args.epochs):
        # train loop
        model.train()

        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            
            is_normal_data = (np.random.randint(3) == 6) # feed normal image with 0% probability
            # is_normal_data = True # feed normal image with 100% probability

            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            if not is_normal_data:
                # mixup process
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            optimizer.zero_grad()

            outs = model(inputs)

            if is_normal_data:
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)
                #loss = criterion(outs, labels) / NUM_ACCUM
            
            else:
                _, preds = torch.max(outs.data, 1)
                loss = mixup_criterion(criterion, outs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            """ # Gradient Accumulation
            if idx % NUM_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad() """

            loss_value += loss.item()

            if is_normal_data:
                matches += (preds == labels).sum().item()
            else:
                matches += (lam * preds.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                
                loss_value = 0
                matches = 0

            wandb.log({"loss": loss})

            if (idx + 1) % 100 == 0:
                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    
                    # for f1
                    targets = []
                    all_predictions = []
                    #

                    val_loss_items = []
                    val_acc_items = []
                    # figure = None

                    for _ in range(4):
                        for val_batch in val_loader:
                            inputs, labels = val_batch
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            outs = model(inputs)
                            preds = torch.argmax(outs, dim=-1)

                            # for f1
                            targets.extend(labels.cpu().numpy())
                            all_predictions.extend(preds.cpu().numpy())
                            #

                            loss_item = criterion(outs, labels).item()
                            acc_item = (labels == preds).sum().item()
                            val_loss_items.append(loss_item)
                            val_acc_items.append(acc_item)
                            '''
                            if figure is None:
                                inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                                inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                                figure = grid_image(
                                    inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                                )
                            '''
                    # calc f1
                    val_f1 = np.mean(f1_score(targets, all_predictions, average=None))

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set) / 4
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        """ 
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth") """
                        best_val_acc = val_acc

                    if val_f1 > best_val_f1:
                        print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..") # best model is created based on its best f1 score
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                        best_val_f1 = val_f1
                    
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1: {best_val_f1:4.2}"
                    )
                    
                    wandb.log({"accuracy": val_acc, "f1-score":val_f1})

                    logger.add_scalar("Val/loss", val_loss, epoch)
                    logger.add_scalar("Val/accuracy", val_acc, epoch)
                    # logger.add_figure("results", figure, epoch)

                    model.train()
                    print()


        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            
            # for f1
            targets = []
            all_predictions = []
            #

            val_loss_items = []
            val_acc_items = []
            # figure = None

            for _ in range(4):
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    # for f1
                    targets.extend(labels.cpu().numpy())
                    all_predictions.extend(preds.cpu().numpy())
                    #

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    '''
                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                    '''
            # calc f1
            val_f1 = np.mean(f1_score(targets, all_predictions, average=None))

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set) / 4
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                """ 
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth") """
                best_val_acc = val_acc

            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..") # best model is created based on its best f1 score
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
            
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1: {best_val_f1:4.2}"
            )
            
            wandb.log({"accuracy": val_acc, "f1-score":val_f1})

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[512, 384], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=200, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
