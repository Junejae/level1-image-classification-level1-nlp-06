import torch.nn as nn
import torch.nn.functional as F
# Junejae's choice of modules
from torchvision import models
import math


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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# Junejae's test model
class MyFcModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

        # initialize
        nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv,stdv)

        """ # Freezing some layers
        count = 0
        for child in self.resnet18.children():
            count += 1
            if count < 6:
                for param in child.parameters():
                    param.requires_grad = False """
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet18(x)
        return x

class MyFcModelDropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.resnet18 = models.resnet50(pretrained=True)
        self.relu = nn.ReLU(True)
        self.output_layer = nn.Linear(in_features=1000, out_features=self.num_classes, bias=True)

        # dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(0.7) for _ in range(5)])

        # initialize
        nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1. / math.sqrt(self.resnet18.fc.weight.size(1))
        self.resnet18.fc.bias.data.uniform_(-stdv,stdv)

        nn.init.xavier_uniform_(self.output_layer.weight)
        stdv = 1. / math.sqrt(self.output_layer.weight.size(1))
        self.output_layer.bias.data.uniform_(-stdv,stdv)

        """ # Freezing some layers
        count = 0
        for child in self.resnet18.children():
            count += 1
            if count < 6:
                for param in child.parameters():
                    param.requires_grad = False """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet18(x)
        feat = self.relu(x)
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                output = self.output_layer(dropout(feat))
            else:
                output += self.output_layer(dropout(feat))
        else:
            output /= len(self.dropouts)

        return output

class MyFcModelDropoutTest(MyFcModelDropout):
    def forward(self, x):
        x = self.resnet18(x)
        feat = self.relu(x)
        output = self.output_layer(feat)
        return output


class MyMlpModelResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(in_features=512, out_features=18)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet18(x)
        return x

class MyMlpModelVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, num_classes),
        )
        
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.vgg16(x)
        return x

class MyMlpModelVGG16bn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.vgg16 = models.vgg16_bn(pretrained=True)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, num_classes),
        )
        
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.vgg16(x)
        return x

class MyMlpModelEffiB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = num_classes
        self.effb1 = models.efficientnet_b1(pretrained=True)
        self.effb1.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.effb1(x)
        return x

