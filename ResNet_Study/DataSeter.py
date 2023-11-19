import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
class Dataseter(Dataset):
    def __init__(self,labelPath:str, transform=None)->None:
        self.labelList = self.labelReader(labelPath)
        self.transform = transform
    def __len__(self)->int:
        return len(self.labelList)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        dataLine = self.labelList[index]
        imgPath = dataLine[0]
        label = dataLine[1]
        img = self.load_image(imgPath)
        class_idx = label
        if self.transform is None:
            transform_list = [transforms.ToTensor()]
            transformer = transforms.Compose(transform_list)
            self.transform = transformer
        tensor = self.transform(img)
        return tensor,class_idx
    def load_image(self, imgPath:str) -> Image.Image:
        "Opens an image via a path and returns it."
        img = Image.open(imgPath)
        return img
    def labelReader(self,labelPath:str)->list[(str,int)]:
        with open(labelPath,'r') as csv_r:
            labelList = csv_r.readlines()
        gtList = []
        for labelLine in labelList:
            labellineData = labelLine.split(',')
            imgPath = labellineData[0]
            imgLabel = int(labellineData[1])
            labelUnit = (imgPath,imgLabel)
            gtList.append(labelUnit)
        return gtList