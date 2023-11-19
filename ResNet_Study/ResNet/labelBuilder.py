import os
import json

def jsonLoader(jsonPath):
    with open(jsonPath) as jL:
        jsonData = json.load(jL)
    return jsonData

if __name__ == '__main__':
    path = "datasets"
    subPaths = ['train_cifar10','test_cifar10']
    jsonPaths = ['annotations/cifar10_train.json','annotations/cifar10_test.json']
    labelPaths = ['train.csv','test.csv']
    for num,_ in enumerate(subPaths):
        fullImagePath = "{}/{}".format(path,subPaths[num])
        fullJsonPath = "{}/{}".format(path,jsonPaths[num])
        jsonFile = jsonLoader(fullJsonPath)
        lenImg = len(jsonFile['images'])
        lenLabel = len(jsonFile['categories'])
        assert lenImg==lenLabel
        csvFile = []
        for i in range(lenImg):
            imageName = jsonFile['images'][i]
            label = jsonFile['categories'][i]
            imagePath = "{}/{}".format(fullImagePath,imageName)
            line = "{},{}\n".format(imagePath,label)
            csvFile.append(line)
        csvPath = "{}/labels/{}".format(path,labelPaths[num])
        with open(csvPath,'w') as csv:
            csv.writelines(csvFile)
        csvFile = []

