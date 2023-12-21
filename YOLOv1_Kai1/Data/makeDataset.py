import os
import random
from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET
random.seed(0)

class MakeDataset:
    def __init__(self,randomSeed:int, dirPath:str):
        self.randomSeed = randomSeed
        self.dirGroup = ['JPEGImages', 'Annotations']
        self.dirPath = dirPath
        self.classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
        self.train_val_ration = 8
    def forward(self,debug=True,savePath=['imgs','labels'],yolov1=True):
        imgsName = self.makeNameList()
        imgList, xmlList = self.makeFullList(imgsName)
        assert len(imgList) == len(xmlList)
        imgsPathList = []
        for num, img in enumerate(imgList):
            boxs, imgWidth, imgHeight = self.readXML(xmlList[num],yolov1=yolov1)
            if(debug):
                self.test_xmlImg(boxs,img)
            bboxs, imgSize = self.boxs2normalize(boxs,imgWidth,imgHeight)
            padedImg = self.paddingImg(img)
            assert (imgSize,imgSize) == padedImg.size
            if(debug):
                self.test_drawImg(bboxs,padedImg)
            else:
                imgsPathList.append(self.saveData(savePath,padedImg,bboxs,448,img))
        if(debug!=True):
            self.makeIndexTXT(imgsPathList)
        print("finish")
    def makeIndexTXT(self,imgsPathList):
        cellList = int(len(imgsPathList) / 10)
        trainRation = self.train_val_ration
        valRation = int((10-trainRation)/2)
        trainPathList = imgsPathList[0:cellList*trainRation]
        valPathList = imgsPathList[cellList*trainRation:cellList*(trainRation+valRation)]
        testPathList = imgsPathList[cellList*(trainRation+valRation):-1]
        self.writeIndex('train.txt',trainPathList)
        self.writeIndex('val.txt',valPathList)
        self.writeIndex('test.txt',testPathList)

    def makeNameList(self) -> [str]:
        random.seed(self.randomSeed)
        imgList = os.listdir("{}/{}".format(self.dirPath,self.dirGroup[0]))
        random.shuffle(imgList)
        return imgList
    def makeFullList(self,imgsName):
        img = []
        xml = []
        for imgName in imgsName:
            img.append("{}/{}/{}".format(self.dirPath,self.dirGroup[0],imgName))
            xml.append("{}/{}/{}".format(self.dirPath,self.dirGroup[1],imgName.replace('jpg','xml')))
        return img, xml
    def readXML(self,xml_path,yolov1=True):
        with open(xml_path, 'r') as f:
            tree = ET.parse(f)
        treeRoot = tree.getroot()
        size = tree.find('size')
        imgWidth = int(size.find('width').text)
        imgHeight = int(size.find('height').text)
        boxs = []
        for obj in treeRoot.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if (cls not in self.classes):
                continue
            elif (yolov1 == True):
                if(difficult==1):
                    continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            x1 = float(xmlbox.find('xmin').text)
            y1 = float(xmlbox.find('ymin').text)
            x2 = float(xmlbox.find('xmax').text)
            y2 = float(xmlbox.find('ymax').text)
            box = [cls_id, x1, y1, x2, y2]
            boxs.append(box)
            if(yolov1 == True):
                break
        return boxs, imgWidth, imgHeight
    @staticmethod
    def boxs2normalize(boxs:[float],width:int,height:int):
        newBoxs = []
        difference = width - height
        for box in boxs:
            x1 = box[1]
            y1 = box[2]
            x2 = box[3]
            y2 = box[4]
            w = x2 - x1
            h = y2 - y1
            x = x1 + w/2
            y = y1 + h/2
            if(difference >= 0):
                y += difference/2
                size = width
            else:
                x += abs(difference)/2
                size = height
            newBox = [box[0],x/size,y/size,w/size,h/size]
            newBoxs.append(newBox)
        return newBoxs, size
    @staticmethod
    def paddingImg(imgPath):
        img = Image.open(imgPath)
        imgSize = img.size
        w = imgSize[0]
        h = imgSize[1]
        difference = abs(w-h)
        if(w == h):
            newImg = img
        elif(w > h):
            # img.show()
            newImgNp = np.zeros([w,w,3],dtype=np.float32)
            imgNp = np.array(img)
            newImgNp[int(difference/2):int(difference/2)+h,0:w] = imgNp
            newImg = Image.fromarray(newImgNp.astype('uint8'))
        else:
            newImgNp = np.zeros([h, h, 3], dtype=np.float32)
            imgNp = np.array(img)
            newImgNp[0:h,int(difference/2):int(difference/2) + w] = imgNp
            newImg = Image.fromarray(newImgNp.astype('uint8'))
        return newImg
    @staticmethod
    def test_drawImg(bboxes:[float],img:Image.Image):
        size = img.size[0]
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            x1 = bbox[1] - (bbox[3]/2)
            y1 = bbox[2] - (bbox[4]/2)
            x2 = bbox[1] + (bbox[3] / 2)
            y2 = bbox[2] + (bbox[4] / 2)
            draw.rectangle(xy=(x1*size,y1*size,x2*size,y2*size))
        img.show()
    @staticmethod
    def test_xmlImg(boxs, imgPath):
        img = Image.open(imgPath)
        for box in boxs:
            draw = ImageDraw.Draw(img)
            draw.rectangle(xy=(box[1],box[2],box[3],box[4]))
        img.show()
    @staticmethod
    def saveData(savePath,img,bbox,imgSize,imgPath):
        fileName = imgPath.split('/')[-1].split('.')[0]
        imgPath = savePath[0]
        labelPath = savePath[1]
        labelFullPath = "{}/{}.txt".format(labelPath,fileName)
        imgFullPath = "{}/{}.jpg".format(imgPath,fileName)
        with open(labelFullPath,'w') as f:
            for box in bbox:
                f.write("{} {} {} {} {}\n".format(box[0],box[1],box[2],box[3],box[4]))
        img = img.resize((imgSize,imgSize))
        img.save(imgFullPath)
        print("finish saving Image:{}\nIn label path:{}, image path:{}".format(fileName,labelFullPath,imgFullPath))
        return imgFullPath
    @staticmethod
    def writeIndex(indexPath,pathsList):
        with open(indexPath,'w') as f:
            for imgPath in pathsList:
                f.write("Data/{}\n".format(imgPath))

if __name__ == '__main__':
    m = MakeDataset(0,"VOC")
    m.forward(False,yolov1=True)