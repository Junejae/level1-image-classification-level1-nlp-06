import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class Resnet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=4096, bias=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.resnet50(x)
        return x


class VGG19(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()

        self.num_classes = num_classes
        self.VGG19 = models.vgg19(pretrained=True)
        self.VGG19.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.VGG19(x)
        return x

# https://www.kaggle.com/iamsdt/ensemble-model-pytorch/notebook
class MyEnsemble(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.model1 = Resnet()
        self.model2 = VGG19()
        self.fc = nn.Linear(in_features=num_classes, out_features=num_classes, bias=True)
        

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        
        total = x1 + x1
        
        output = self.fc(total)
        return output

