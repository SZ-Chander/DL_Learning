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
        if(trans == None):
            transform_list = [transforms.ToTensor()]
            transformer = transforms.Compose(transform_list)
            self.transform = transformer
        else:
            self.transform = trans
