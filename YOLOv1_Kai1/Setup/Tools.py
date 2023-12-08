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