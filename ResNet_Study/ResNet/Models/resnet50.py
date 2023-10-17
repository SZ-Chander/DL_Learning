import torch
from torch import nn
from .resTools import Bottleneck

class ResNet50(nn.Module):
    def __init__(self,num_classes):
        super(ResNet50,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1)
        )
        self.conv2_1 = Bottleneck(64,64,1,True)
        self.conv2_2 = Bottleneck(256,64,1,False)
        self.conv2_3 = Bottleneck(256,64,1,False)

        self.conv3_1 = Bottleneck(256,128,2,True)
        self.conv3_2 = Bottleneck(512,128,1,False)
        self.conv3_3 = Bottleneck(512,128,1,False)
        self.conv3_4 = Bottleneck(512,128,1,False)

        self.conv4_1 = Bottleneck(512,256,2,True)
        self.conv4_2 = Bottleneck(1024,256,1,False)
        self.conv4_3 = Bottleneck(1024,256,1,False)
        self.conv4_4 = Bottleneck(1024,256,1,False)
        self.conv4_5 = Bottleneck(1024,256,1,False)
        self.conv4_6 = Bottleneck(1024,256,1,False)

        self.conv5_1 = Bottleneck(1024,512,2,True)
        self.conv5_2 = Bottleneck(2048,512,1,False)
        self.conv5_3 = Bottleneck(2048,512,1,False)

        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self,x):
        x = self.conv1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x