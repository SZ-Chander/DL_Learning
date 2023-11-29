a = [1,2,3,4,5,6,7,8,9,10]
box = []
boxes = []
for n,i in enumerate(a):
    box.append(i)
    if((n+1)%5 == 0):
        boxes.append(box)
        box=[]
print(boxes)