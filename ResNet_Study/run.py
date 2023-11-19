import time
import torch
from torchvision.transforms import transforms
from DataSeter import Dataseter
from ResNet.Models.resnet18 import ResNet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Train:
    def __init__(self,train_dataLoader,val_dataLoader,net,lossFunction,optimizer,epoch,device):
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.net = net
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        self.epoch = epoch
        self.device = device
    def forward(self):
        device = self.device
        model = self.net.to(device)
        val_data_iter = iter(self.val_dataLoader)
        val_image, val_label = val_data_iter.__next__()
        val_image = Variable(val_image)
        val_label = Variable(val_label)
        optimizer = self.optimizer
        lossFunction = self.lossFunction
        totalTimeStart = time.time()
        for epoch in range(self.epoch):
            print("epoch{} started.".format(epoch + 1))
            timeStart = time.time()
            running_loss = 0.00
            for step,data in enumerate(self.train_dataLoader,0):
                inputTensor,labelsTensor = data
                inputTensor = inputTensor.to(device)
                labelsTensor = labelsTensor.to(device)

                inputTensor,labelsTensor = Variable(inputTensor),Variable(labelsTensor)
                optimizer.zero_grad()
                outputsTensor = model(inputTensor)
                loss = lossFunction(outputsTensor,labelsTensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if step % 250 == 249:
                    with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                        len_data = 0
                        correct = 0.00
                        for data in testloader:
                            val_image, val_label = data
                            len_data += val_label.size(0)
                            val_image = Variable(val_image).to(device)
                            val_label = val_label.to(device)
                            outputsTesnor = model(val_image)
                            _, predict_y = torch.max(outputsTesnor.data, 1)
                            correct += (predict_y == val_label).sum()
                        accuracy = correct / len_data
                        print('[%d, %5d] train_loss: %.3f  val_accuracy: %.3f' %
                              (epoch + 1, step + 1, loss.item(), accuracy))

            print('epoch %d cost %3f sec' % (epoch + 1, time.time() - timeStart))
        print('Finished Training')
        totalTime = time.time() - totalTimeStart
        try:
            print("training {} epochs cost {} sec".format(self.epoch,totalTime))
        except:
            pass
        try:
            save_path = 'models/resnet18.pth'
            torch.save(model.state_dict(), save_path)
            print("saved moudels in {}".format(save_path))
        except:
            print("Error to save moudel")

if __name__ == '__main__':

    train_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomGrayscale(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    test_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         ])

    trainset = Dataseter(labelPath='datasets/labels/train.csv',transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = Dataseter(labelPath='datasets/labels/test.csv',transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = ResNet18(10)
    device = torch.device('mps')
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train = Train(train_dataLoader=trainloader,
                  val_dataLoader=testloader,
                  net=net,
                  lossFunction=lossFunction,
                  optimizer=optimizer,
                  epoch=50,
                  device=device)
    train.forward()