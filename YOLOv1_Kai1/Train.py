from Setup.Timer import TimeLogMission
import torch
from MyDataset import MyDataset
import datetime
import torch.nn as nn
from Moudle.YOLOv1 import YOLOv1Kai
from Setup.Tools import Tools
from torch.utils.data import DataLoader
from Moudle.Loss import YoloV1Loss
# ==============import package==========
timeLogMission = TimeLogMission.timeLogMission
# =========Global Variable and Method=========

class Train:
    def __init__(self,batch_size:int,lr:float,weight:float,epoch:int,start_epoch:int,device:str,dataset_dir:str,save_path:str,num_workers:int,pretrain:bool,pretrain_path:str,model:nn.Module,image_txt_path:str,label_dir_path:str,classes:list[str],val_txt_path:str,logFile:str,epoch_range:int,setupPath:str):
        self.batch_size = batch_size
        self.Lr = lr
        self.weight = weight
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.device = device
        self.dataset_dir = dataset_dir
        self.save_path = save_path
        self.num_workers = num_workers
        self.pretrain = pretrain
        self.pretrain_path = pretrain_path
        self.model = model
        self.image_txt_path = image_txt_path
        self.label_dir_path = label_dir_path
        self.classes = classes
        self.val_txt_path = val_txt_path
        self.logFile = logFile
        self.epoch_range = epoch_range
        self.setupPath = setupPath
    def forward(self):
        timeStart = datetime.datetime.now()
        train_dataset = MyDataset(imgTxtPath=self.image_txt_path,labelDirPath=self.label_dir_path,classes=self.classes)
        train_dataLoader = DataLoader(train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers)
        val_dataset = MyDataset(imgTxtPath=self.val_txt_path, labelDirPath=self.label_dir_path, classes=self.classes)
        val_dataLoader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)
        if (self.pretrain == True):
            model = torch.load(self.pretrain_path)
        else:
            model = self.model
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.Lr, momentum=0.9, weight_decay=self.weight)
        totalEpoch = 0
        self.writeLog("Train start at {}\nArguments by {}\n{}\n".format(datetime.datetime.now(), self.setupPath, '='*30))
        for epoch in range(self.start_epoch, self.epoch+1):
            totalEpoch += 1
            self.train(model=model,train_loader=train_dataLoader,device=self.device,optimizer=optimizer,epoch=epoch,totalEpoch=self.epoch,
                       missionName="Epoch{}".format(epoch),logPath=self.logFile)
            if ((epoch + 1) % self.epoch_range == 0):
                self.save_model(model=model, epoch=epoch, save_path=self.save_path)
        timeEnd = datetime.datetime.now()
        self.writeLog("Train End at {}, consumes {}\nTrain {} epochs, Average time spent {}\n".format(timeEnd,timeEnd-timeStart,self.epoch-self.start_epoch,(timeEnd-timeStart)/(self.epoch-self.start_epoch)))
        self.save_model(model=model,epoch=self.epoch,save_path=self.save_path)
        print("Train Mission Complete")
    def writeLog(self,mess) -> None:
        with open(self.logFile,'a+') as log:
            log.write(mess)
    @staticmethod
    def save_model(model:nn.Module,epoch,save_path) -> None:
        modelName = "epoch{}.pt".format(epoch)
        save_dir = "{}/{}".format(save_path,modelName)
        torch.save(model, save_dir)
    @staticmethod
    @timeLogMission
    def train(model,train_loader,device,optimizer,epoch,totalEpoch) -> str:
        model.train()
        avg_loss = 0.0
        startMess = "Train epoch{} start at {}\n{}".format(epoch,datetime.datetime.now(),'='*30)
        print(startMess)
        totalIter = len(train_loader)
        totalEpochMess = ""
        for num,(imgs,labels) in enumerate(train_loader):
            labels = labels.permute(0,3,1,2)
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            loss = YoloV1Loss().loss(in_pred=pred,labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * num + loss.item())/(num+1)
            if(num % 20 == 0):
                epochMess = "Epoch {}/{} | Iter {}/{} | training loss={:.3f}, avg_loss={:.3f}".format(epoch, totalEpoch,num,totalIter,loss.item(),avg_loss)
                print(epochMess)
                totalEpochMess += "{}\n".format(epochMess)
        logMess = "{}\n{}".format(startMess,totalEpochMess)
        return logMess

def startTrain(jsonData:dict, model:nn.Module, setupPath:str):
    batch_size = jsonData['batch_size']
    LR = jsonData['LR']
    weight = jsonData['weight']
    epoch = jsonData['epoch']
    start_epoch = jsonData['start_epoch']
    device = jsonData['device']
    dataset_dir = jsonData['dataset_dir']
    save_path = jsonData['save_path']
    num_workers = jsonData['num_workers']
    pretrain = jsonData['pretrain']
    pretrain_path = jsonData['pretrain_path']
    image_txt_path = jsonData['image_txt_path']
    label_dir_path = jsonData['label_dir_path']
    val_txt_path = jsonData['val_txt_path']
    classes = jsonData['GL_CLASSES']
    logPath = jsonData['log_file']
    epoch_range = jsonData['epoch_range']

    Train(
        batch_size=batch_size,lr=LR,weight=weight,epoch=epoch,start_epoch=start_epoch,
        device=device,dataset_dir=dataset_dir,save_path=save_path,num_workers=num_workers,
        pretrain=pretrain,pretrain_path=pretrain_path,model=model,image_txt_path=image_txt_path,
        label_dir_path=label_dir_path,classes=classes,val_txt_path=val_txt_path,logFile=logPath,
        epoch_range=epoch_range, setupPath=setupPath
    ).forward()

    print('Train finished')

if __name__ == '__main__':
    setupPath = 'Arguments/Train_from_epoch0.json'
    jsonData = Tools().readJson(setupPath)
    GL_CLASSES = jsonData['GL_CLASSES']
    model = YOLOv1Kai(GL_CLASSES)
    startTrain(jsonData, model,setupPath)