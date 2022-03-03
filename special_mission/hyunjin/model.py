import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import argparse


#parser = argparse.ArgumentParser()
#parser.add_argument('--model_name', type=str, default='model')
#args = parser.parse_args()
    
class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=True)
        
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=18, bias=True)
        
        # initialize
        nn.init.xavier_uniform_(self.resnet50.fc.weight)
        stdv = 1. / math.sqrt(self.resnet50.fc.weight.size(1))
        self.resnet50.fc.bias.data.uniform_(-stdv,stdv)
        
    def forward(self, x):
        x = self.resnet50(x)
        return x
    

class SOTA(nn.Module):
    def __init__(self, num_classes=18, device='cuda'):
        super().__init__()
        
        model_path = os.path.join('./model/sota_v2.pth')
        self.model = MyCustomModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
                nn.Linear(in_features=1280, out_features=2048, bias=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.efficientnet(x)
        return x
    
        
class Resnet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.resnet50 = models.resnet18(pretrained=True)
        self.resnet50.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=2048, bias=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.resnet50(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.vgg19 = models.vgg16(pretrained=True)
        self.vgg19.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.vgg19(x)
        return x

# https://www.kaggle.com/iamsdt/ensemble-model-pytorch/notebook
class MyEnsemble(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.modelA = Resnet()
        self.modelB = VGG19()
        self.modelC = EfficientNet()
        self.fc1 = nn.Linear(in_features=num_classes, out_features=num_classes, bias=True)
        

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        
        total = x1 + x2 #+ x3
        
        output = self.fc1(total)
        return output

