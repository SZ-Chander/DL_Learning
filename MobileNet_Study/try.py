from torchvision.models import mobilenet_v3_large
import torch

net = torch.load(mobilenet_v3_large())
torch.save(net,'checkpoints/a.pt')