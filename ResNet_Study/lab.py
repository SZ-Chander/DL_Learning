#  声明：本代码并非自己编写，由他人提供
import torch
import torchvision
import torchvision.transforms as transforms
import ssl

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from ResNet.Models.resnet18 import ResNet18
from DataSeter import Dataseter

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
#-----数据准备-----
trainset = Dataseter(labelPath='datasets/labels/train.csv',transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

testset = Dataseter(labelPath='datasets/labels/test.csv',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#-----模型准备-----
net = ResNet18(10)
device = torch.device('mps')
net = net.to(device)
# cpu_device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


if __name__ == '__main__':
    totalTime = 0
    for epoch in range(10):
        print("epoch{} is started.".format(epoch+1))
        timestart = time.time()
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device) #1
            labels = labels.to(device) #2
            # print("inputs is in CPU? {}\nlabels is in CPU? {}".format(inputs.is_cpu,labels.is_cpu))
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:
                print('[%d ,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        print('epoch %d cost %3f sec' % (epoch + 1, time.time()-timestart))
        totalTime += time.time()-timestart

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.__next__()
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(Variable(images).to(device))
    _, predicted = torch.max(outputs.data,1)
    print('Predicted:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        variable = Variable(images).to(device)
        outputs = net(variable)
        # print(type(outputs))
        outputs = outputs.to(device)
        _, predicted = torch.max(outputs.data, 1)
        # print(type(predicted))
        total += labels.size(0)
        labels = labels.to(device)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        variable = Variable(images).to(device)
        outputs = net(variable)
        outputs = outputs.to(device)
        labels = labels.to(device)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("一共 100 epoch 花费 {} 秒".format(totalTime))