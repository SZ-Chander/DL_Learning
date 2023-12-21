import time
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from MyDataset import MyDataset
from Moudle.Loss import YoloV1Loss
import torchvision
from Moudle.YOLOv1 import YOLOv1Kai

class Val:
    def __init__(self,val_dataloader:DataLoader,super_trust:float,super_iou:float,classes:[str],device:str,gridSize):
        self.val_dataloader = val_dataloader
        self.batchsize = val_dataloader.batch_size
        self.super_trust = super_trust
        self.super_iou = super_iou
        self.classes = classes
        self.device = device
        self.gridSize = gridSize
    def forward(self,model:nn.Module) -> str:
        model.eval()
        totalIou = 0.0
        totalIter = 0
        model.to(self.device)
        with torch.no_grad():
            calculateIOU = YoloV1Loss.calculateIOU
            batch2xyxyBox = YoloV1Loss.batch2xyxyBox
            for num,(imgs, _, gtBox, _1) in enumerate(self.val_dataloader):
                gtList = self.transGTBox(gtBox)
                preds = model(imgs.to(self.device))
                for n_batch, pred in enumerate(preds):
                    clsDict = {}
                    gtLines = gtList[n_batch]
                    for row in range(self.gridSize):
                        for column in range(self.gridSize):
                            box = torch.stack(batch2xyxyBox(pred,row=row,column=column,num_grid=self.gridSize,start=0))
                            trust, cls = self.getTrustAndCls(pred,row=row,column=column,start=0,superTrust=self.super_trust,len_classes=len(self.classes))
                            if(trust != -1):
                                if(cls not in clsDict):
                                    clsDict[cls] = []
                                clsDict[cls].append((box,trust))
                            box = torch.stack(batch2xyxyBox(pred, row=row, column=column, num_grid=self.gridSize, start=5))
                            trust, cls = self.getTrustAndCls(pred, row=row, column=column, start=5,superTrust=self.super_trust, len_classes=len(self.classes))
                            if (trust != -1):
                                if (cls not in clsDict):
                                    clsDict[cls] = []
                                clsDict[cls].append((box, trust))
                    clsDict = self.mutiClsNMS(clsDict,self.super_iou)
                    for gt in gtLines:
                        gt_cls = int(gt[0])
                        # totalIter += 1
                        if(gt_cls in clsDict):
                            bestIOU = 0.0
                            for dataLine in clsDict[gt_cls]:
                                totalIter += 1
                                gt_box = torch.stack((gt[1],gt[2],gt[3],gt[4]))
                                gt_box = gt_box.to(self.device)
                                iou = calculateIOU(dataLine,gt_box)
                                bestIOU += iou
                                # if(iou > bestIOU):
                                #     bestIOU = iou
                            totalIou += bestIOU
        mess = "\nEpoch{}"
        return mess + " average IOU = {}\n".format(totalIou/totalIter)

    @staticmethod
    def mutiClsNMS(dataDict:dict,super_iou) -> dict:
        newDataDict = {}
        for cls in dataDict:
            boxes = []
            trusts = []
            cls_datas = dataDict[cls]
            for cls_data in cls_datas:
                boxes.append(cls_data[0])
                trusts.append(cls_data[1])
            if(len(boxes)!=1):
                newBoxes = []
                keys = torchvision.ops.nms(torch.stack(boxes),torch.stack(trusts),super_iou)
                for key in keys:
                    newBoxes.append(boxes[int(key)])
                newDataDict[cls] = newBoxes
            else:
                newDataDict[cls] = boxes
        return newDataDict
    @staticmethod
    def getTrustAndCls(in_pred:torch.Tensor,row:int,column:int,start:int,superTrust,len_classes):
        trust = in_pred[4, row, column]
        if(trust <= superTrust):
            return -1, -1
        else:
            clses = in_pred[10+start:len_classes+10+start,row,column]
            return trust, int(torch.argmax(clses))
    @staticmethod
    def transGTBox(gt):
        boxes = []
        batchSize = len(gt[0])
        num_line = int(len(gt)/5)
        for num in range(batchSize):
            batch = []
            for line in range(num_line):
                cls = gt[0+line*5][num]
                x = gt[1+line*5][num]
                y = gt[2+line*5][num]
                w = gt[3+line*5][num]
                h = gt[4+line*5][num]
                box = (cls,(x-w/2).float(),(y-h/2).float(),(x+w/2).float(),(y+h/2).float())
                batch.append(box)
            boxes.append(batch)
        return boxes

if __name__ == '__main__':
    cls = [
        "person","bird","cat","cow","dog","horse","sheep","aeroplane","bicycle","boat",
        "bus","car","motorbike","train","bottle","chair","diningtable","pottedplant","sofa","tvmonitor"
        ]
    val_dataset = MyDataset(imgTxtPath='Data/test.txt', labelDirPath='Data/labels', classes=cls)
    val_dataLoader = DataLoader(val_dataset, batch_size=4, num_workers=0, shuffle=True)
    val = Val(val_dataloader=val_dataLoader,super_trust=0.4,super_iou=0.5,classes=cls,device='cpu',gridSize=7)
    # model = torch.load("checkpoints/20231129/epoch49.pt")
    model = YOLOv1Kai(cls)
    model.load_state_dict(torch.load("checkpoints/231216/epoch97.pt"))
    t1 = time.time()
    a = val.forward(model=model)
    t2 = time.time()
    print(t2-t1)
    print(a.format(85))