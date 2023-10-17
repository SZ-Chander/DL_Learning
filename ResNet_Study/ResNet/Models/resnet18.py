import torch
from torch import nn
from .resTools import BasicBlock


class ResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.conv2_1 = BasicBlock(64,64,1)
        self.conv2_2 = BasicBlock(64,64,1)

        self.conv3_1 = BasicBlock(64,128,2,downsample=True)
        self.conv3_2 = BasicBlock(128,128,1)

        self.conv4_1 = BasicBlock(128,256,2,downsample=True)
        self.conv4_2 = BasicBlock(256,256,1)

        self.conv5_1 = BasicBlock(256,512,2,downsample=True)
        self.conv5_2 = BasicBlock(512,512,1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        x = self.conv1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

