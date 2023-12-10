import time
import torch
from MyDataSet import MyDataset
import datetime
import torch.nn as nn
from Models.MyYOLOv1 import YOLOv1
from Tools.Tools import Tools
from torch.utils.data import DataLoader
from Models.Loss import Yolov1Loss
from Tools.CitedUtil import Util
import numpy as np

class Train:
    def __init__(self,batch_size:int,lr:float,weight:float,epoch:int,start_epoch:int,device:str,dataset_dir:str,save_path:str,num_workers:int,pretrain:bool,pretrain_path:str,model:nn.Module,image_txt_path:str,label_dir_path:str,classes:list[str],val_txt_path:str,logFile:str,epoch_range:int):
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
    def forward(self):
        train_dataset = MyDataset(imgTxtPath=self.image_txt_path,labelDirPath=self.label_dir_path,classes=self.classes)
        train_dataLoader = DataLoader(train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers)
        val_dataset = MyDataset(imgTxtPath=self.val_txt_path,labelDirPath=self.label_dir_path,classes=self.classes)
        val_dataLoader = DataLoader(val_dataset, 1, num_workers=0)
        # val_dataLoader = DataLoader(val_dataset,self.batch_size,num_workers=self.num_workers)
        num_train = len(train_dataset.imgPathList)
        if(self.pretrain == True):
            model = torch.load(self.pretrain_path)
        else:
            model = self.model
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.Lr, momentum=0.9, weight_decay=self.weight)
        num_totalEpoch = 0
        for epoch in range(self.start_epoch,self.epoch+1):
            num_totalEpoch += 1
            t_start = time.time()
            self.trainStep(model=model,train_loader=train_dataLoader,device=self.device,
                           optimizer=optimizer,logPath=self.logFile,epoch=epoch,totalEpoch=self.epoch,
                           batch_size=self.batch_size,num_train=num_train)
            t_end = time.time()

            print("Training consumes %.2f second\n" % (t_end - t_start))
            with open(self.logFile, "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t_end - t_start))

            mess = self.valStep(model=model, val_loader=val_dataLoader, device=self.device, epoch=epoch,
                                val_range=self.epoch_range)
            print(mess)
            if((epoch+1) % self.epoch_range == 0):
                self.save_model(model=model,epoch=epoch,save_path=self.save_path)
        self.save_model(model=model,epoch=num_totalEpoch,save_path=self.save_path)
        print("Mission complete")

    @staticmethod
    def trainStep(model,train_loader,device,optimizer,logPath,epoch,totalEpoch,batch_size,num_train):
        model.train() # 启动训练模式
        # avg_metric = 0.  # 平均评价指标
        avg_loss = 0.  # 平均损失数值
        logFile = open(logPath,'a+')
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
        logFile.write(localtime)
        logFile.write("\n======================training epoch %d======================\n"%epoch)
        for num,(imgs,labels,check_path,_) in enumerate(train_loader):
            # print("p0 {} from {}".format(labels.shape,check_path))
            labels = labels.view(batch_size, 7, 7, -1)  # 一维升 7x7
            labels = labels.permute(0,3,1,2)  # 转置矩阵
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            loss = Yolov1Loss().loss(in_pred=pred,labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = (avg_loss * num + loss.item())/(num+1)  #计算平均loss
            if num % 20 == 0:  # 根据打印频率输出log信息和训练信息
                print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                      (epoch, totalEpoch, num, num_train // batch_size, loss.item(), avg_loss))
                logFile.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                               (epoch, totalEpoch, num, num_train // batch_size, loss.item(), avg_loss))
                logFile.flush()
        # log end
        logFile.close()
    def gtb2bbox(self,gtbox)->list[list[float]]:
        box = []
        boxes = []
        for n, i in enumerate(gtbox):
            box.append(float(i))
            if ((n + 1) % 5 == 0):
                newbox = self.xywh2xyxy(box)
                boxes.append(newbox)
                box = []
        return boxes

    @staticmethod
    def xywh2xyxy(box):
        x1 = box[1] - box[3] / 2
        y1 = box[2] - box[4] / 2
        x2 = box[1] + box[3] / 2
        y2 = box[2] + box[4] / 2
        newbox = [x1, y1, x2, y2, 1.0, box[0]]
        return newbox

    def valStep(self,model: nn.Module, val_loader: DataLoader, device: str, epoch: int, val_range: int):
        util = Util()
        model.eval()
        avg_iou = 0.0
        num_val = 0
        logFile = open(self.logFile, 'a+')
        with torch.no_grad():
            for num, (imgs, labels, _, GTBox) in enumerate(val_loader):
                boxes = self.gtb2bbox(GTBox)
                imgs = imgs.to(device)
                pred = model(imgs).cpu()
                pred = pred.squeeze(dim=0).permute(1, 2, 0)
                pred_bbox = util.labels2bbox(pred)
                for box in np.array(boxes):
                    for p_box in pred_bbox:
                        if(p_box[4] <= 0.4):
                            continue
                        iou = Yolov1Loss().calculate_iou(box, p_box)
                        avg_iou += iou
                        num_val += 1
            title = "======================validate epoch {}======================".format(epoch)
            mess = "Average IOU in epoch{} - epoch{} is {}\n".format(epoch - val_range + 1, epoch, avg_iou / num_val)    #
            fullMess = "{}\n{}".format(title,mess)
            logFile.write(fullMess)
            logFile.flush()
            logFile.close()
            return mess
    @staticmethod
    def save_model(model:nn.Module,epoch,save_path):
        modelName = "epoch{}.pt".format(epoch)
        save_dir = "{}/{}".format(save_path,modelName)
        torch.save(model,save_dir)


def startTrain(jsonData:dict,model:nn.Module):
    print("train start")
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
        epoch_range=epoch_range
    ).forward()

    print('Train finished')

if __name__ == '__main__':
    setupPath = 'Arguments/Train_start.json'
    jsonData = Tools().readJson(setupPath)
    GL_CLASSES = jsonData['GL_CLASSES']
    model = YOLOv1(GL_CLASSES)
    startTrain(jsonData,model)
