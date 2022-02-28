import torch
import numpy as np
import random
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
def train(data_loader, model, loss_fn, optm, EPOCH):
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

            y_pred = model(imgs)
            loss = loss_fn(y_pred, labels)
            targets.extend(labels.cpu().numpy())
            all_preds.extend(y_pred.argmax(dim=-1).detach().cpu().numpy())
            
            optm.zero_grad()
            loss.backward()
            optm.step()

#             if i % len(data_loader) == 0:
        print("epoch: {} | Loss: {:.4f} | Acc: {:.4f}".format(epoch, loss.data, accuracy_score(targets, all_preds)))
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
