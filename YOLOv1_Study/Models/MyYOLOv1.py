import torch.nn as nn
import torchvision.models as tvmodel

class YOLOv1(nn.Module):
    def __init__(self,classes:list[str]):
        super(YOLOv1, self).__init__()
        self.classes = classes
        resnet = tvmodel.resnet34(pretrained=True)
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024,1024,3,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,7*7*(5*2+len(classes))),
            nn.Sigmoid()
        )

    def forward(self,inputs):
        x = self.resnet(inputs)
        x = self.Conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.Conn_layers(x)
        x = x.reshape(-1, (5 * 2 + len(self.classes)), 7, 7)  # 记住最后要reshape一下输出数据
        return x