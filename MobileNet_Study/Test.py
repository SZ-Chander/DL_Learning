import torch.nn as nn
import datetime
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from torchvision.models import mobilenet_v3_large
from torch.autograd import Variable

class Test:
    def __init__(self):
        self.num_workers = 0
        self.batchSize = 64
        self.device = 'mps'
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def forward(self,model):
        val_dataset = torchvision.datasets.CIFAR100(root="Data/cifar100", train=False, transform=self.val_transform,
                                                    download=False)
        val_dataLoader = DataLoader(dataset=val_dataset, batch_size=self.batchSize, shuffle=True,
                                    num_workers=self.num_workers)
        total_acc = 0.00
        total_Iter = 0

        with torch.no_grad():
            model.eval()
            model.to(self.device)
            for num, datas in enumerate(val_dataLoader):
                imgs, labels = datas
                imgs = imgs.to(self.device)
                preds = model(imgs)
                for batch_num, pred in enumerate(preds):
                    total_Iter += 1
                    output = torch.argmax(pred).data
                    label = labels[batch_num].data
                    if(label.item() == output.item()):
                        total_acc += 1
            print(total_acc/total_Iter)


if __name__ == '__main__':
    model_ft = torch.load("checkpoints/epoch7.pt")
    test = Test()
    test.forward(model_ft)