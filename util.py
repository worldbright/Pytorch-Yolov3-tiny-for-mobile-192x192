import torch
import tqdm
import numpy as np

def load_weights(model):
    with open('./yolov3-tiny.conv.15', 'rb') as f:
        version = np.fromfile(f, dtype=np.int32, count=3)
        seen = np.fromfile(f, dtype=np.int64, count=1)
        weights = np.fromfile(f, dtype=np.float32)
    ptr = 0
    idxs = list(range(0, 9, 2)) + [9]
    for idx in idxs:
        bn = model.base_modules[idx][1]
        # batchnorm weights
        nb = bn.bias.numel()
        bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
        ptr += nb
        # Weight
        bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
        ptr += nb
        # Running Mean
        bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
        ptr += nb
        # Running Var
        bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
        ptr += nb

        # conv weights
        conv = model.base_modules[idx][0]
        nw = conv.weight.numel()
        conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
        ptr += nw
    return
    for idx in [0, 1]:
        bn = model.main_modules1[idx][1]
        # batchnorm weights
        nb = bn.bias.numel()
        bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
        ptr += nb
        # Weight
        bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
        ptr += nb
        # Running Mean
        bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
        ptr += nb
        # Running Var
        bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
        ptr += nb

        # conv weights
        conv = model.main_modules1[idx][0]
        nw = conv.weight.numel()
        conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
        ptr += nw

def nms(prediction, conf_thres, nms_thres):
    prediction[..., :4] = get_xxyy_from_xywh(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        image_pred = image_pred[image_pred[:, 3] >= 0]
        image_pred = image_pred[image_pred[:, 2] >= 0]
        image_pred = image_pred[image_pred[:, 1] >= 0]
        image_pred = image_pred[image_pred[:, 0] >= 0]
        if not image_pred.size(0): continue
        score = image_pred[:, 4]
        image_pred = image_pred[(-score).argsort()]
        keep_boxes = []
        while image_pred.size(0):
            large_overlap = torch.cat([torch.tensor([True]), get_bbox_iou(image_pred[0, :4].unsqueeze(0), image_pred[1:, :4]) > nms_thres])
            keep_boxes += [image_pred[0]]
            image_pred = image_pred[~large_overlap]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output

def get_xxyy_from_xywh(coord):
    ret = coord.new(coord.shape)
    ret[..., 0] = coord[..., 0] - coord[..., 2] / 2
    ret[..., 1] = coord[..., 1] - coord[..., 3] / 2
    ret[..., 2] = coord[..., 0] + coord[..., 2] / 2
    ret[..., 3] = coord[..., 1] + coord[..., 3] / 2
    return ret

def get_iou_wh(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def get_bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    if not x1y1x2y2:
        # xywh 를 x1y1x2y2로 변환
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 겹치는 좌표 구하고
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 겹치는 넓이
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # 합친 넓이
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union = b1_area + b2_area - inter_area + 1e-16
    # 계산
    iou = inter_area / union
    return iou

def parsing_target(pred_box, target, anchors, threshold, device):
    batch_num = pred_box.size(0)
    grid_num = pred_box.size(2)

    # 구할 값 들
    obj_mask = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.bool, device=device)
    no_obj_mask = torch.ones(batch_num, 3, grid_num, grid_num, dtype=torch.bool, device=device)
    iou_score = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.float, device=device)
    tx = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.float, device=device)
    ty = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.float, device=device)
    tw = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.float, device=device)
    th = torch.zeros(batch_num, 3, grid_num, grid_num, dtype=torch.float, device=device)

    # 타겟에서 바운딩박스 좌표 그리드 단위로 변환 및 추출
    target_boxes = target[:, 1:5] * grid_num
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # 타겟과 앵커박스의 베스트 iou 구함
    ious = torch.stack([get_iou_wh(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)
    # 타겟 batch_num, xywh 분리
    b = target[:, :1].long().view(-1)
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # 마스크 설정
    obj_mask[b, best_ious_idx, gj, gi] = 1
    no_obj_mask[b, best_ious_idx, gj, gi] = 0
    # threshold 확인 및 noobj 0으로 설정
    for i, anchor_ious in enumerate(ious.t()):
        no_obj_mask[b[i], anchor_ious > threshold, gj[i], gi[i]] = 0

    # xy좌표 그리드를 기준으로 설정
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()
    # wh를 prediction결과 형식에 맞춤
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx, 1] + 1e-16)
    # iou 계산
    iou_score[b, best_ious_idx, gj, gi] = get_bbox_iou(pred_box[b, best_ious_idx, gj, gi], target_boxes, False)
    # obj마스크
    tconf = obj_mask.float()

    return iou_score, obj_mask, no_obj_mask, tx, ty, tw, th, tconf