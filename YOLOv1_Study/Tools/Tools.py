import json

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
