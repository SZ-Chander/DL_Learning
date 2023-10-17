import ResNet.Models.resnet18 as resnet18
from ResNet.Models.resnet50 import ResNet50
import torch

if __name__ == '__main__':
    net = resnet18.ResNet18(1000)
    # net = ResNet50(1000)
    net.eval()
    # print(net)
    r_input = torch.randn(1,3,64,64)
    y = net(r_input)
    # device = torch.device('cpu')
    # t = net.to(device)
    print(y)
