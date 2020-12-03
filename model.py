import torch
import torch.nn as nn
import numpy as np
from util import parsing_target
def getConv_bn(inp_ch, out_ch, kernel_size, stride, padding, bias):
    return nn.Sequential(nn.Conv2d(inp_ch, out_ch, kernel_size, stride, padding, bias=bias),
                         nn.BatchNorm2d(out_ch, momentum=0.03, eps=1E-4))
def getConv_bn_ac(inp_ch, out_ch, kernel_size, stride, padding, bias):
    return nn.Sequential(nn.Conv2d(inp_ch, out_ch, kernel_size, stride, padding, bias=bias),
                         nn.BatchNorm2d(out_ch, momentum=0.03, eps=1E-4),
                         nn.LeakyReLU(0.1, inplace=True))

class df_yoloLayer(nn.Module):
    def __init__(self, anchors, img_size, device):
        super(df_yoloLayer, self).__init__()
        self.anchors = anchors
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.iou_thresh = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.device = device
    def forward(self, x, targets):
        batch_num = x.size(0)
        grid_size = x.size(2)
        # 처리하기 편한 view 형태로 변환 (마지막 차원에 prediction 결과)
        prediction = (x.view(batch_num, 3, 5, grid_size, grid_size).permute(0,1,3,4,2).contiguous())

        # x, y, w, h, object점수 얻기
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_dog = torch.sigmoid(prediction[..., 4])

        # stride = grid 한 칸 픽셀 수
        stride = self.img_size / grid_size

        # grid x y 인덱스 표현
        grid_x = torch.arange(grid_size, dtype=torch.float, device = self.device).repeat(grid_size, 1).view(
            1, 1, grid_size, grid_size)
        grid_y = torch.arange(grid_size, dtype=torch.float, device = self.device).repeat(grid_size, 1).t().view(
            1, 1, grid_size, grid_size)
        # 예측한 w, h를 그리드 단위로 표현 하기 위한 변수
        # scaled_anchors : 앵커박스 w, h를 그리드 단위로 변환
        # anchor box의 단위 설정
        scaled_anchors = torch.as_tensor([(aw / stride, ah / stride) for aw, ah in self.anchors], dtype=torch.float, device=self.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, 3, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, 3, 1, 1))
        # x, y 좌표를 그리드 좌측상단 좌표를 기준으로 옮겨주고,
        # w, h 를 exp씌워서 위에서 구한 단위를 곱해줌
        # 최종 예측 값 완성
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w).clamp(max=1E3) * anchor_w
        pred_boxes[..., 3] = torch.exp(h).clamp(max=1E3) * anchor_h
        pred = (pred_boxes.view(batch_num, -1, 4) * stride,
                pred_dog.view(batch_num, -1, 1))
        output = torch.cat(pred, -1)
        if targets is None:
            return output, 0
        # 타겟에서 정보 파싱해서
        iou_score, obj_mask, no_obj_mask, tx, ty, tw, th, tconf = parsing_target(pred_boxes, targets, scaled_anchors, self.iou_thresh, self.device)
        # 전체적인 로스 구하기
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        loss_conf_obj = self.bce_loss(pred_dog[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_dog[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_layer = loss_bbox + loss_conf

        #텐서보드 기록
        conf50 = (pred_dog > 0.5).float()
        iou50 = (iou_score > 0.5).float()
        iou75 = (iou_score > 0.75).float()
        detected_mask = conf50 * tconf
        conf_obj = pred_dog[obj_mask].mean()
        conf_no_obj = pred_dog[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }

        return output, loss_layer

class df_basedYolo(nn.Module):
    def __init__(self, img_size, device):
        super(df_basedYolo, self).__init__()
        self.base_modules = nn.ModuleList()
        self.main_modules1 = nn.ModuleList()
        self.main_modules2 = nn.ModuleList()

        # yolo layer전에 주로 사용하는 conv, maxpool layer들
        self.base_modules.append(getConv_bn_ac(3, 16, 3, 1, 1, False))              #0
        self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  #1
        self.base_modules.append(getConv_bn_ac(16, 32, 3, 1, 1, False))             #2
        self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  #3
        self.base_modules.append(getConv_bn_ac(32, 64, 3, 1, 1, False))             #4
        self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  #5
        self.base_modules.append(getConv_bn_ac(64, 128, 3, 1, 1, False))            #6
        self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  #7
        self.base_modules.append(getConv_bn_ac(128, 256, 3, 1, 1, False))           #8
        #self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  #9
        self.base_modules.append(getConv_bn_ac(256, 512, 3, 1, 1, False))           #10
        #self.base_modules.append(nn.ZeroPad2d((0, 1, 0, 1)))                        #11
        #self.base_modules.append(nn.MaxPool2d(kernel_size=2, stride=1, padding=0))  #12
        #self.base_modules.append(getConv_bn_ac(512, 1024, 3, 1, 1, False))          #13

        # 첫번째 yolo layer
        #self.main_modules1.append(getConv_bn_ac(1024, 256, 1, 1, 0, False))
        self.main_modules1.append(getConv_bn_ac(512, 256, 1, 1, 0, False))
        self.main_modules1.append(getConv_bn_ac(256, 512, 3, 1, 1, False))
        self.main_modules1.append(getConv_bn(512, 15, 1, 1, 0, False))
        self.yolo1 = df_yoloLayer([(81//2, 82//2), (135//2, 169//2), (344//2, 319//2)], img_size, device)

        # 두번째 yolo layer
        # route layer -4 생략
        self.main_modules2.append(getConv_bn_ac(256, 128, 1, 1, 0, False))
        #self.main_modules2.append(nn.Upsample(scale_factor=2, mode="nearest"))
        # route layer -1, 8 생략
        self.main_modules2.append(getConv_bn_ac(128+256, 256, 3, 1, 1, False))
        self.main_modules2.append(getConv_bn(256, 15, 1, 1, 0, False))
        self.yolo2 = df_yoloLayer([(10//2,14//2),(23//2,27//2),(37//2,58//2)], img_size, device)

        self.yolo_layers = [self.yolo1, self.yolo2]

    def forward(self, x, targets=None):
        saved_out_base = {}
        saved_out_module1 = {}
        out = x
        for i, module in enumerate(self.base_modules):
            out = module(out)
            saved_out_base[i] = out

        for i, module in enumerate(self.main_modules1):
            out = module(out)
            saved_out_module1[i] = out
        yolo_out1, loss1 = self.yolo1(out, targets)

        out = saved_out_module1[0]
        out = self.main_modules2[0](out)
        out = torch.cat((out, saved_out_base[8]), 1)
        out = self.main_modules2[1](out)
        out = self.main_modules2[2](out)
        yolo_out2, loss2 = self.yolo2(out, targets)
        loss = loss1 + loss2
        yolo_output = torch.cat([yolo_out1, yolo_out2], 1).detach().cpu()
        return yolo_output if targets is None else (loss, yolo_output)