from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from Moudle.Loss import YoloV1Loss
import torchvision
from Moudle.YOLOv1 import YOLOv1Kai
import numpy as np

class Pred:
    def __init__(self,super_trust:float,super_iou:float,classes:[str],device:str,gridSize):
        self.super_trust = super_trust
        self.super_iou = super_iou
        self.classes = classes
        self.device = device
        self.gridSize = gridSize
        self.colors = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
                       (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
                       (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
                       (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0)]
        self.transform = transforms.ToTensor()
    def forward(self,model:nn.Module,imgPath) -> str:
        model.eval()
        img = Image.open(imgPath)
        imgName = imgPath.split('/')[-1]
        imgNp = np.array(img)
        imgTensor = self.transform(imgNp)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = imgTensor.to(self.device)
        model.to(self.device)
        with torch.no_grad():
            batch2xyxyBox = YoloV1Loss.batch2xyxyBox
            preds = model(imgTensor)
            for n_batch, pred in enumerate(preds):
                clsDict = {}
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
                self.drawImg(img=img,boxesDict=clsDict,imgName=imgName)
                print(clsDict)
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
    def drawImg(self,img:Image.Image,boxesDict:dict,imgName:str):
        imgDraw = ImageDraw.Draw(img)
        for num_dict in boxesDict:
            boxesTensor = boxesDict[num_dict]
            for boxTensor in boxesTensor:
                if(len(boxTensor) == 4):
                    box = (boxTensor[0]*448,boxTensor[1]*448,boxTensor[2]*448,boxTensor[3]*448)
                    imgDraw.rectangle(xy=box,outline=self.colors[num_dict])
                    ft = ImageFont.truetype("Arial.ttf",20)
                    imgDraw.text(xy=(box[0], box[1] - 21), text=self.classes[num_dict], fill=self.colors[num_dict],font=ft)
        img.save("Pred/pred_{}".format(imgName))

if __name__ == '__main__':
    cls = [
        "person","bird","cat","cow","dog","horse","sheep","aeroplane","bicycle","boat",
        "bus","car","motorbike","train","bottle","chair","diningtable","pottedplant","sofa","tvmonitor"
        ]
    val = Pred(super_trust=0.4,super_iou=0.5,classes=cls,device='mps',gridSize=7)
    model = YOLOv1Kai(cls)
    model.load_state_dict(torch.load("checkpoints/231216/epoch97.pt"))
    val.forward(model=model,imgPath="Data/imgs/2010_006035.jpg")