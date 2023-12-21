import os
p = "/Users/szchandler/Desktop/DLPythonCode/DL_Learning/YOLOv1_Kai1/Data/labels"
for txt_p in os.listdir(p):
    if(txt_p.split('.')[-1] == 'txt'):
        with open("{}/{}".format(p,txt_p)) as f:
            txtData = txt_p
            box = []
            for i in txtData:
                if(len(i.replace('\n','').replace(' ','')) > 5):
                    box.append(i)
            if(len(box) != 1):
                print(txt_p)
                print(box)
