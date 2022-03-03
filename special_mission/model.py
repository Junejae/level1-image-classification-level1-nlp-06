import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Resnet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.resnet50 = models.resnet18(pretrained=True)
        self.resnet50.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=2048, bias=True),
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
        self.fc1 = nn.Linear(in_features=num_classes, out_features=num_classes, bias=True)
        

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        
        total = x1 + x2
        
        output = self.fc1(total)
        return output

