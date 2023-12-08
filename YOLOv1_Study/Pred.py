import numpy as np
import torch.nn as nn
import torchvision.transforms
from torch.utils.data import DataLoader
from Tools.CitedUtil import Util
from PIL import Image,ImageDraw,ImageFont
import torch
from Models.Loss import Yolov1Loss
from MyDataSet import MyDataset

class Pred:
    def __init__(self,imgPath:str,modelPath:str,device:str):
        self.imgPath = imgPath
        self.modelPath = modelPath
        self.device = device
        self.classes = [
        "person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat",
        "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable",
        "pottedplant", "sofa", "tvmonitor"]
        self.colors = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0)]

    def forward(self):
        img, boxes = self.startPred()
        draw_boxes = self.boxes2List(boxes)
        draw_boxes = self.checkTruth(draw_boxes,0.5)
        if(len(draw_boxes) == 0):
            return False
        else:
            self.drawImg(img,draw_boxes)
            return True

    def checkTruth(self,boxes:[[float]],trustLine:float)->[[float]]:
        newBox = []
        for box in boxes:
            if(box[4] >= trustLine):
                newBox.append(box)
        return newBox


    def drawImg(self,img:Image.Image,boxes:[[float]]):
        imgSize = img.size[0]
        imgDraw = ImageDraw.Draw(img)
        for box in boxes:
            p1 = (box[0]*imgSize,box[1]*imgSize)
            p2 = (box[2]*imgSize,box[3]*imgSize)
            print(p1,p2)
            cls = int(box[5])
            imgDraw.rectangle(xy=(p1,p2),outline=self.colors[cls])
            # font = ImageFont.truetype()
            imgDraw.text(xy=(p1[0]-10,p1[1]-10),text=self.classes[cls],fill=self.colors[cls])
        img.show()



    def startPred(self)->(Image.Image, np.ndarray):
        model = torch.load(self.modelPath)
        model = model.to(self.device)
        img = Image.open(self.imgPath)
        labels = self.predsigleImg(img,model,self.device)
        boxes = self.pred2Boxes(labels)
        return img, boxes

    # @staticmethod
    # def xywh2xyxy(x,y,w,h,t,cls)->list[float]:
    #     half_w = w/2
    #     half_h = h/2
    #     x1 = x-half_w
    #     y1 = y-half_h
    #     x2 = x+half_w
    #     y2 = y+half_h
    #     # return [x,y,w,h,t,cls]
    #     return [x1,y1,x2,y2,t,cls]

    def boxes2List(self,boxes:np.ndarray)->list[list[float]]:
        newBoxList = []
        for box in boxes:
            # x,y,w,h,t,cls
            x = float(box[0])
            y = float(box[1])
            # w = float(box[2])
            x2 = float(box[2])
            y2 = float(box[3])
            # h = float(box[3])
            t = float(box[4])
            cls = float(box[5])
            # newBoxList.append(self.xywh2xyxy(x,y,w,h,t,cls))
            newBoxList.append([x,y,x2,y2,t,cls])
        return newBoxList

    @staticmethod
    def predsigleImg(img:Image.Image,model,device)->torch.Tensor:
        imgNp = np.array(img,dtype=np.float32)
        transform = torchvision.transforms.ToTensor()
        imgTensor = transform(imgNp)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = imgTensor.to(device)
        pred = model(imgTensor)
        return pred

    @staticmethod
    def pred2Boxes(pred:torch.Tensor)->np.ndarray:
        util = Util()
        pred = pred.cpu().detach()
        pred = pred.squeeze(dim=0).permute(1, 2, 0)
        pred_bbox = util.labels2bbox(pred)
        return pred_bbox

if __name__ == '__main__':
    imgPath = "Data/img/2010_000227.jpg"
    modelPath = "checkpoints/20231129/epoch49.pt"
    device = 'mps'
    pred = Pred(imgPath,modelPath,device)
    success = pred.forward()
    if(success):
        print("finish")
    else:
        print("error")
