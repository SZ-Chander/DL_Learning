from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from Moudle.Loss import YoloV1Loss
import torchvision
from Moudle.YOLOv1 import YOLOv1Kai
import numpy as np

class Pred:
    def __init__(self,super_trust:float,super_iou:float,classes:[str],device:str,gridSize:int,imgTrainSize=(448,448)):
        self.super_trust = super_trust
        self.super_iou = super_iou
        self.classes = classes
        self.device = device
        self.gridSize = gridSize
        self.imgTrainSize = imgTrainSize
        self.colors = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
                       (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
                       (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
                       (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0)]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(imgTrainSize)
            ]
        )
    def forward(self,model:nn.Module,imgPath) -> str:
        model.eval()
        img = Image.open(imgPath)
        img = self.expand2square(img,0)
        imgLongSize = img.size[0]
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
                self.drawImg(img=img,boxesDict=clsDict,imgName=imgName,imgLongSize=imgLongSize)
                print(clsDict)
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
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
    def drawImg(self,img:Image.Image,boxesDict:dict,imgName:str,imgLongSize:int):
        # imgTrainSize = self.imgTrainSize[0]
        # proportion = imgLongSize / imgTrainSize
        imgDraw = ImageDraw.Draw(img)
        for num_dict in boxesDict:
            boxesTensor = boxesDict[num_dict]
            for boxTensor in boxesTensor:
                if(len(boxTensor) == 4):
                    box = (boxTensor[0]*imgLongSize,boxTensor[1]*imgLongSize,boxTensor[2]*imgLongSize,boxTensor[3]*imgLongSize)
                    imgDraw.rectangle(xy=box,outline=self.colors[num_dict],width=10)
                    ft = ImageFont.truetype("Arial.ttf",100)
                    imgDraw.text(xy=(box[0], box[1] - 100), text=self.classes[num_dict], fill=self.colors[num_dict],font=ft)
        img.save("Pred/pred_{}".format(imgName))

if __name__ == '__main__':
    cls = [
        "person","bird","cat","cow","dog","horse","sheep","aeroplane","bicycle","boat",
        "bus","car","motorbike","train","bottle","chair","diningtable","pottedplant","sofa","tvmonitor"
        ]
    val = Pred(super_trust=0.3,super_iou=0.5,classes=cls,device='mps',gridSize=7, imgTrainSize=(448,448))
    model = YOLOv1Kai(cls)
    model.load_state_dict(torch.load("checkpoints/YOLOv1Kai_VOC2012_Epoch100.pt"))
    val.forward(model=model,imgPath="Pred/neko.jpg")