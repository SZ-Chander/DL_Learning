import json
import numpy as np
class Tools:
    @staticmethod
    def readJson(jsonPath:str)->dict:
        with open(jsonPath, 'r') as f:
            jsonData = json.load(f)
        return jsonData
    @staticmethod
    def removeReturnFromList(inputList:list[str])->list[str]:
        retrunList = []
        for line in inputList:
            retrunList.append(line.replace('\n',''))
        return retrunList
    @staticmethod
    def readTxt(txtPath:str)->list[str]:
        with open(txtPath) as f:
            txt = f.readlines()
        return txt
    @staticmethod
    def labelStr2Float(labelLines:list[str])->list[float]:
        labelBbox = []
        for labelLine in labelLines:
            labelStrList = labelLine.split(' ')
            for labelStr in labelStrList:
                if(labelStr != '\n'):
                    labelBbox.append(float(labelStr))
        return labelBbox
    @staticmethod
    def bbox2labels(bbox:[float],gridNum:int,len_classes:int) -> np.ndarray:
        gridSize = 1.0 / gridNum
        labelNp = np.zeros((7,7,10+len_classes),dtype=np.float32)
        for i in range(len(bbox) // 5):
            gridColumn = int(bbox[i * 5 + 1] // gridSize)
            gridRow = int(bbox[i * 5 + 2] // gridSize)
            px = bbox[i * 5 + 1] / gridSize - gridColumn
            py = bbox[i * 5 + 2] / gridSize - gridRow
            labelNp[gridRow, gridColumn,0:5] = np.array([px, py, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
            labelNp[gridRow, gridColumn,5:10] = np.array([px, py, bbox[i*5+3], bbox[i*5+4], 1])
            labelNp[gridRow, gridColumn, 10 + int(bbox[i*5+0])] = 1
        return labelNp
