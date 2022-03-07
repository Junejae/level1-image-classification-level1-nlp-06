import wandb

import torch
import numpy as np
import random
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    
def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    train_set = Subset(dataset, indices=train_idx)
    val_set = Subset(dataset, indices=valid_idx)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    return train_loader, val_loader


def k_fold_train(k, dataset, model, loss_fn, criterion_fn, optm, BATCH_SIZE, EPOCH, num_workers, 
                 is_wandb_logging=False, accumulation_steps=2, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    skf = StratifiedKFold(n_splits=k)
    
    # for Early stopping
    counter = 0
    best_loss = np.inf
    
    ## Train process
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, dataset.multi_class_labels)):
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, BATCH_SIZE, num_workers)
        n = len(train_loader)
        m = len(val_loader)

        for epoch in range(EPOCH):
            targets = []
            all_preds = []
            # train process
            model.train()
            for i, train_batch in enumerate(train_loader):
                imgs, labels = train_batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                outs = model(imgs)
                preds = torch.argmax(outs, dim=-1)
                loss = loss_fn(outs, labels)
                loss.backward()
                
                # Gradient Accumulation
                if (i+1) % accumulation_steps == 0:
                    optm.step()
                    optm.zero_grad()
            
                targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
#             if is_wandb_logging:
            wandb.log({"train_loss": np.mean(f1_score(targets, all_preds, average=None)), "train_acc": accuracy_score(targets, all_preds)})
            
            targets = []
            all_preds = []
            # eval process
            with torch.no_grad():
                model.eval()
                for i, eval_batch in enumerate(val_loader):
                    imgs, labels = eval_batch
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    outs = model(imgs)
                    preds = torch.argmax(outs, dim=-1)
                    
                    targets.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            f1 = np.mean(f1_score(targets, all_preds, average=None))
            acc = accuracy_score(targets, all_preds)
            print("F1 Loss: {:.4f} | Accuracy: {:.4f}".format(f1, acc))
#             if is_wandb_logging:
            wandb.log({"eval_loss": f1, "eval_acc": acc})
                
            if f1 < best_loss:
                best_loss = f1
            else:
                counter += 1
            if counter > patience:
                print('Early Stopping...')
                break










def train(data_loader, model, loss_fn, optm, EPOCH, is_wandb_logging=False, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## Train process
    model.train()
    for epoch in range(EPOCH):
        # image shape : [8, 3, 512, 384]
        targets = []
        all_preds = []
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = Variable(imgs).to(device)
            labels = Variable(labels).to(device)

            outs = model(imgs)
            preds = torch.argmax(outs, dim=-1)
            loss = loss_fn(outs, labels)
            
            loss.backward()
            optm.step()
            optm.zero_grad()

            targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            
#             if i % len(data_loader) == 0:
        acc = accuracy_score(targets, all_preds)
        print("epoch: {} | Loss: {:.4f} | Acc: {:.4f}".format(epoch, np.mean(f1_score(targets, all_preds, average=None)), acc))
        
        if is_wandb_logging:
            wandb.log({"loss": f1_score(targets, all_preds, average=None), "accuracy": acc})
    print('done!')
    
    
def eval(eval_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## Eval process
    targets = []
    all_preds = []
    with torch.no_grad():
        model.eval()
        for i, (imgs, labels) in enumerate(eval_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            y_pred = model(imgs).argmax(dim=-1)
            targets.extend(labels.cpu().numpy())
            all_preds.extend(y_pred.cpu().numpy())
    #         print(y_pred, labels)
    #         tensor([8, 2, 2, 2, 8, 2, 8, 7], device='cuda:0') tensor([11,  2,  1,  2,  8,  2,  8,  7], device='cuda:0')
    print('done!')
    
    
    ## evaluate Metric
    print("Accuracy: {:.4f}".format( accuracy_score(targets, all_preds)) )
    print("F1 Loss: {:.4f}".format( np.mean(f1_score(targets, all_preds, average=None)) ))
