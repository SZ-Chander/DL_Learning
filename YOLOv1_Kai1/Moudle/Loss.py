import torch
class YoloV1Loss:
    def __init__(self):
        self.lambdaCoord = 5.0
        self.lambdaNoobj = 0.5
        self.num_grid = 7

    def loss(self, in_pred:torch.Tensor, labels:torch.Tensor):
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小
        for batch in range(n_batch):
            for n in range(self.num_grid):
                for m in range(self.num_grid):
                    if(labels[batch,4,m,n] == 1):
                        bbox1_pred_xyxy = self.batch2xyxyBox(in_pred[batch],row=m,column=n,num_grid=self.num_grid,start=0)
                        bbox2_pred_xyxy = self.batch2xyxyBox(in_pred[batch],row=m,column=n,num_grid=self.num_grid,start=5)
                        bbox_gt_xyxy = self.batch2xyxyBox(labels[batch],row=m,column=n,num_grid=self.num_grid,start=0)
                        iou_box1 = self.calculateIOU(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou_box2 = self.calculateIOU(bbox2_pred_xyxy, bbox_gt_xyxy)
                        if(iou_box1 >= iou_box2):
                            start = 0
                            iou1 = iou_box1
                            iou2 = iou_box2
                        else:
                            start = 5
                            iou1 = iou_box2
                            iou2 = iou_box1
                        coor_loss += self.lambdaCoord * self.loss_line1_2(in_pred[batch],labels[batch],row=m,colum=n,start=start)
                        obj_confi_loss += self.loss_line3_4(in_pred[batch],iou1,row=m,column=n,start=start)
                        noobj_confi_loss += self.lambdaNoobj * self.loss_line3_4(in_pred[batch],iou2,row=m,column=n,start=abs(start-5))
                        class_loss += self.loss_line5(in_pred[batch],labels[batch],row=m,column=n)
                    else:
                        noobj_confi_loss += self.lambdaNoobj * self.loss_noobj(in_pred[batch],row=m,column=n)
        loss = coor_loss + obj_confi_loss + class_loss + noobj_confi_loss
        # print("coor_loss={}, obj_confi_loss={}, class_loss={}, noobj_confi_loss={}".format(coor_loss,obj_confi_loss,class_loss,noobj_confi_loss))
        return loss / n_batch
    @staticmethod
    def batch2xyxyBox(in_pred:torch.Tensor,row:int,column:int,num_grid:int,start:int) -> tuple:
        # a = (in_pred[0+start, row, column] + column) / num_grid - in_pred[2+start, row, column] / 2
        bbox1_pred_xyxy = ((in_pred[0+start, row, column] + column) / num_grid - in_pred[2+start, row, column] / 2,
                           (in_pred[1+start, row, column] + row) / num_grid - in_pred[3+start, row, column] / 2,
                           (in_pred[0+start, row, column] + column) / num_grid + in_pred[2+start, row, column] / 2,
                           (in_pred[1+start, row, column] + row) / num_grid + in_pred[3+start, row, column] / 2)
        return bbox1_pred_xyxy
    @staticmethod
    def calculateIOU(pred_box:tuple,gt_box:tuple) -> torch.Tensor:
        if (pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1] or gt_box[2] <= gt_box[0] or gt_box[3] <= gt_box[1]):
            return torch.tensor(0.0)
        coincidentBox = [0.0, 0.0, 0.0, 0.0]
        coincidentBox[0] = (max(pred_box[0],gt_box[0]))
        coincidentBox[1] = (max(pred_box[1], gt_box[1]))
        coincidentBox[2] = (min(pred_box[2], gt_box[2]))
        coincidentBox[3] = (min(pred_box[3], gt_box[3]))
        w = max(coincidentBox[2] - coincidentBox[0], 0)
        h = max(coincidentBox[3] - coincidentBox[1], 0)
        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        area_coincident = w * h
        iou = area_coincident / (area_pred + area_gt - area_coincident + 1e-6)
        return iou
    @staticmethod
    def loss_line1_2(in_pred:torch.Tensor,label:torch.Tensor,row:int,colum:int,start:int):
        xy = torch.sum((in_pred[0+start:2+start,row,colum] - label[0+start:2+start,row,colum])**2)
        wh = torch.sum((in_pred[2+start:4+start,row,colum].sqrt() - label[2+start:4+start,row,colum].sqrt())**2)
        return xy + wh
    @staticmethod
    def loss_line3_4(in_pred:torch.Tensor,iou:float,row:int,column:int,start:int):
        return (in_pred[4+start,row,column] - iou) ** 2
    @staticmethod
    def loss_line5(in_pred:torch.Tensor,label:torch.Tensor,row:int,column:int) -> torch.Tensor:
        cls = torch.sum((in_pred[10:,row,column] - label[10:,row,column]) ** 2 )
        return cls
    @staticmethod
    def loss_noobj(in_pred:torch.Tensor,row:int,column:int):
        return torch.sum(in_pred[[4,9],row,column] ** 2)

if __name__ == '__main__':
    pred_sample = "../sample/Pred_Iter371.pt"
    label_sample = "../sample/Label_Iter371.pt"
    pred = torch.load(pred_sample).cpu()
    labels = torch.load(label_sample).cpu()
    loss = YoloV1Loss().loss(in_pred=pred,labels=labels)
    print(loss)