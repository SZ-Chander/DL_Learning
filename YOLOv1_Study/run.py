import time
import torch

if __name__ == '__main__':
    data = torch.load("/Users/szchandler/Desktop/DLPythonCode/DL_Learning/YOLOv1_Kai1/Moudle/sample_2010_000227.pt")
    t = 0.0
    num = 100000
    t1 = time.time()
    for i in range(num):
        t1 = time.time()
        a = data[0]
        t2 = time.time()
        t += (t2 - t1)
    print("索引方法进行{}次运行共花费{}秒，平均{}秒一次".format(num,t,t/num))
    t1 = time.time()
    t = 0.0
    for i in range(num):
        t1 = time.time()
        a = data.take(0)
        t2 = time.time()
        t += (t2 - t1)
    print("take方法进行{}次运行共花费{}秒，平均{}秒一次".format(num, t, t / num))