import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from Tools.Tools import Tools

class MyDataset(Dataset):
    def __init__(self,imgTxtPath:str,labelDirPath:str,classes:list[str],imgType:str='jpg', trans=None):
        self.labelDirPath = labelDirPath
        self.imgType = imgType
        self.imgPathList = Tools.removeReturnFromList(self.readTxt(imgTxtPath))
        self.lenDataset = len(self.imgPathList)
        self.classes = classes
        if(trans == None):
            transform_list = [transforms.ToTensor()]
            transformer = transforms.Compose(transform_list)
            self.transform = transformer
        else:
            self.transform = trans
    def __getitem__(self, index)->(str,torch.Tensor,np.ndarray):
        imgPath = self.imgPathList[index].replace('\n','')
        imgNumpy = np.array(Image.open(imgPath))
        imgTensor = self.transform(imgNumpy)
        labelPath = self.imgPath2LabelPath(imgPath,self.imgType)
        labelStr = self.readTxt(labelPath)
        labelFloatList = self.labelStr2Float(labelStr)
        labelNumpy = self.convert_bbox2labels(labelFloatList)
        return imgTensor,labelNumpy,imgPath,labelFloatList
    def __len__(self)->int:
        return self.lenDataset

    def labelStr2Float(self,labelLines:list[str])->list[float]:
        labelBbox = []
        for labelLine in labelLines:
            labelStrList = labelLine.split(' ')
            for labelStr in labelStrList:
                if(labelStr != '\n'):
                    labelBbox.append(float(labelStr))
        return labelBbox

    def readTxt(self,txtPath:str)->list[str]:
        with open(txtPath) as f:
            txt = f.readlines()
        return txt
    def imgPath2LabelPath(self,imgPath:str,imgType:str)->str:
        labelName = imgPath.split('/')[-1].replace(imgType,'txt').replace('\n','')
        labelPath = "{}/{}".format(self.labelDirPath,labelName)
        return labelPath

    def convert_bbox2labels(self,bbox)->np.ndarray:
        # code cite from https://github.com/lavendelion/YOLOv1-from-scratch?search=1
        """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
        gridsize = 1.0 / 7
        labels = np.zeros((7, 7, 5 * 2 + len(self.classes)),dtype=np.float32)  # 注意，此处需要根据不同数据集的类别个数进行修改
        for i in range(len(bbox) // 5):
            gridx = int(bbox[i * 5 + 1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
            gridy = int(bbox[i * 5 + 2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
            # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
            gridpx = bbox[i * 5 + 1] / gridsize - gridx
            gridpy = bbox[i * 5 + 2] / gridsize - gridy
            # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
            labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
            labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
            labels[gridy, gridx, 10 + int(bbox[i * 5])] = 1
        labels = labels.reshape(1, -1)
        return labels

if __name__ == '__main__':
    GL_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                  'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
                  'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    md = MyDataset('Data/train.txt','Data/labels',GL_CLASSES,'jpg')
    a, b, c = md.__getitem__(0)
    print(a)
    print(b)
    print(c)