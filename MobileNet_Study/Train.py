import torch.nn as nn
import datetime
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from Tools.TimeLoger import TimeLogMission
from torch.autograd import Variable
timeLogMission = TimeLogMission.timeLogMission

class Train:
    def __init__(self,model,LR):
        self.num_workers = 4
        self.batchSize = 64
        self.device = 'mps'
        self.epoch = 100
        self.model = model
        self.LR = LR
        self.missionName = "MobileNet_v3"
        self.logPath = "log/231226.txt"
        self.save_path = "checkpoints"
        self.optimizer = optim.Adam(model.parameters(), lr=LR)
        self.lossFunction = nn.CrossEntropyLoss()
        self.epoch_range = 1
        self.start_epoch = 0
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def setNumworkers(self,num_workers:int):
        self.num_workers = num_workers
    def setDevice(self,device:str):
        self.device = device
    def setBatchSize(self,batchSize:int):
        self.batchSize = batchSize
    def setTrainTransform(self,train_transform):
        self.train_transform = train_transform
    def setValTransform(self,val_transform):
        self.val_transform = val_transform
    def setEpoch(self,epoch:int):
        self.epoch = epoch
    def setModel(self,model:torch.nn.Module):
        self.model = model
    def setOptimizer(self,optimizer):
        self.optimizer = optimizer
    def setLossFunction(self,lossFunction):
        self.lossFunction = lossFunction
    def setEpochRange(self,epoch_range):
        self.epoch_range = epoch_range

    def forward(self):
        train_dataset = torchvision.datasets.CIFAR100(root="Data/cifar100", train=True, transform=self.train_transform,download=False)
        val_dataset = torchvision.datasets.CIFAR100(root="Data/cifar100", train=False, transform=self.val_transform,download=False)
        train_dataLoader = DataLoader(dataset=train_dataset, batch_size=self.batchSize, shuffle=True, num_workers=self.num_workers)
        val_dataLoader = DataLoader(dataset=val_dataset, batch_size=self.batchSize, shuffle=True, num_workers=self.num_workers)
        model = self.model.to(self.device)
        timeStart = datetime.datetime.now()
        for epoch in range(self.epoch):
            self.train(model=model,train_loader=train_dataLoader,device=self.device,optimizer=self.optimizer,
                       epoch=epoch,totalEpoch=self.epoch,missionName=self.missionName,logPath=self.logPath,
                       lossFunction=self.lossFunction)
            if((epoch + 1) % self.epoch_range == 0):
                self.val(model=model,val_loader=val_dataLoader,device=self.device,epoch=epoch,
                         totalEpoch=self.epoch,missionName=self.missionName,logPath=self.logPath)
                self.save_model(model=model,epoch=epoch,save_path=self.save_path)
        timeEnd = datetime.datetime.now()
        self.writeLog("Train End at {}, consumes {}\nTrain {} epochs, Average time spent {}\n".format(timeEnd,timeEnd - timeStart,self.epoch - self.start_epoch, ( timeEnd - timeStart) / (self.epoch - self.start_epoch)))
        self.save_model(model=model, epoch=self.epoch, save_path=self.save_path)
        print("Train Mission Complete")
    def writeLog(self,mess) -> None:
        with open(self.logPath,'a+') as log:
            log.write(mess)
    @staticmethod
    @timeLogMission
    def train(model,train_loader,device,optimizer,epoch,totalEpoch,lossFunction)->str:
        avg_loss = 0.00
        totalIter = len(train_loader)
        totalEpochMess = ""
        startMess = "Train epoch{} start at {}\n{}".format(epoch, datetime.datetime.now(), '=' * 30)
        model.train()
        print(startMess)
        for num, data in enumerate(train_loader):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs, labels = Variable(imgs), Variable(labels)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = lossFunction(preds, labels)
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * num + loss.item())/(num+1)
            if(num % 20 == 0):
                epochMess = "Epoch {}/{} | Iter {}/{} | training loss={:.3f}, avg_loss={:.3f}".format(epoch, totalEpoch, num, totalIter, loss.item(), avg_loss)
                print(epochMess)
                totalEpochMess += "{}\n".format(epochMess)
        logMess = "{}\n{}".format(startMess, totalEpochMess)
        return logMess
    @staticmethod
    @timeLogMission
    def val(model,val_loader,device,epoch,totalEpoch)->str:
        total_correct = 0.00
        totalIter = len(val_loader)
        total_Iter = 0.00
        startMess = "Valid epoch{} start at {}\n{}".format(epoch, datetime.datetime.now(), '=' * 30)
        with torch.no_grad():
            model.eval()
            print(startMess)
            for num, datas in enumerate(val_loader):
                imgs, labels = datas
                imgs = imgs.to(device)
                preds = model(imgs)
                for batch_num, pred in enumerate(preds):
                    total_Iter += 1
                    output = torch.argmax(pred).data
                    label = labels[batch_num].data
                    if(label.item() == output.item()):
                        total_correct += 1
            accuracy = total_correct / totalIter
            valMess = "Epoch{}/{} | mean accuracy={:.3f}".format(totalEpoch,epoch,accuracy)
            print(valMess)
        logMess = "{}\n{}".format(startMess, valMess)
        return logMess

    @staticmethod
    def save_model(model: nn.Module, epoch, save_path) -> None:
        modelName = "epoch{}.pt".format(epoch)
        save_dir = "{}/{}".format(save_path, modelName)
        torch.save(model, save_dir)


if __name__ == '__main__':
    LR = 1e-4
    model_ft = torch.load("checkpoints/epoch2.pt")
    num_ftrs = model_ft.classifier[3].in_features
    model_ft.classifier[3] = torch.nn.Linear(num_ftrs, 100)
    train_instance = Train(model_ft,LR)
    train_instance.forward()