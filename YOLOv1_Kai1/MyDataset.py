import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from Setup.Tools import Tools
import torchvision.transforms as transforms
# ==============import package==========

class MyDataset(Dataset):
    def __init__(self,imgTxtPath:str,labelDirPath:str,classes:list[str],imgType:str='jpg', trans=None):
        self.labelDirPath = labelDirPath
        self.imgType = imgType
        self.imgPathList = Tools.removeReturnFromList(Tools.readTxt(imgTxtPath))
        self.lenDataset = len(self.imgPathList)
        self.classes = classes
        self.len_classes = len(self.classes)
        if(trans == None):
            transform_list = [transforms.ToTensor()]
            transformer = transforms.Compose(transform_list)
            self.transform = transformer
        else:
            self.transform = trans
    def __len__(self)->int:
        return self.lenDataset
    def __getitem__(self, index) -> (torch.Tensor, np.ndarray):
        imgPath = self.imgPathList[index].replace('\n','')
        imgNumpy = np.array(Image.open(imgPath))
        imgTensor = self.transform(imgNumpy)
        labelPath = self.imgPath2LabelPath(imgPath, self.imgType)
        labelStr = Tools.readTxt(labelPath)
        labelFloatList = Tools.labelStr2Float(labelStr)
        labelNp = Tools.bbox2labels(labelFloatList,7,self.len_classes)
        return imgTensor, labelNp, labelFloatList, labelPath

    def imgPath2LabelPath(self,imgPath:str, imgType:str)->str:
        labelName = imgPath.split('/')[-1].replace(imgType,'txt').replace('\n','')
        labelPath = "{}/{}".format(self.labelDirPath,labelName)
        return labelPath
