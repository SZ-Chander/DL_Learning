import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,stride:int,downsample=False):
        super(BasicBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,stride,bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.downsample = downsample

    def forward(self,x):
        out = self.left(x)
        residual = x if self.downsample==False else self.shortcut(x)
        out += residual
        return F.relu(out)

class Bottleneck(nn.Module):
    def __init__(self,in_channel:int,place:int,stride:int,downsample=False,expansion=4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=place,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=place,out_channels=place,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=place,out_channels=place*self.expansion,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(place*expansion)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=place*self.expansion,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(place*expansion)
        )
    def forward(self,x):
        out = self.left(x)
        residual = x if self.downsample==False else self.shortcut(x)
        out += residual
        return F.relu(out)