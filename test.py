import cv2
import time
import os

import torch
import torch.utils.data
import numpy as np
import tqdm

from model import df_basedYolo
import utilData
import util



def evaluate(model, image_path, target_path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device, output=False):
    model.eval()

    dataSet = utilData.ListDataset(image_path, target_path, augment=False, img_size=image_size)
    dataLoader = torch.utils.data.DataLoader(dataSet,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=dataSet.collate_fn)

    labels = []
    correct = 0
    error = 0
    entire_time = 0
    if output and not os.path.isdir('./dog_dataset/eval/result_image'): os.mkdir('./dog_dataset/eval/result_image')
    for _, images, targets in tqdm.tqdm(dataLoader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        labels.extend(targets[:, 1].tolist())
        targets[:, 1:] = util.get_xxyy_from_xywh(targets[:, 1:])
        targets[:, 1:] *= image_size

        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = util.nms(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time
        if output:
            for i, path in enumerate(_):
                img = cv2.imread(path)
                h, w, a = img.shape
                if h > w: pad = [0, 0, (h-w)//2, (h-w)-((h-w)//2)]
                else: pad = [(w-h)//2,(w-h)-((w-h)//2), 0, 0]
                img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
                splited_path = os.path.split(path)
                if outputs[i] is None:
                    if targets[i, 1:].sum() == 0:
                        correct += 1
                    else:
                        error += 1
                        print('outputnone error',path)
                    cv2.imwrite('./dog_dataset/eval/result_image/' + splited_path[1], img)
                    continue
                for box in outputs[i]:
                    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                if targets[i, 1:].sum() == 0:
                    error += 1
                    print('targetnone error', path)
                    cv2.imwrite('./dog_dataset/eval/result_image/' + splited_path[1], img)
                    continue
                ious = util.get_bbox_iou(outputs[i], targets[i:i+1, 1:], True)
                for iou in ious:
                    if iou >= iou_thres: correct += 1/len(ious)
                    else         :
                        error += 1/len(ious)
                        print(iou, path)
                cv2.imwrite('./dog_dataset/eval/result_image/'+splited_path[1], img)
        else:
            for i, path in enumerate(_):
                if outputs[i] is None:
                    if targets[i, 1:].sum() == 0:
                        correct += 1
                    else:
                        error += 1
                    continue
                ious = util.get_bbox_iou(outputs[i], targets[i:i + 1, 1:], True)
                for iou in ious:
                    if iou >= iou_thres: correct += 1 / len(ious)
                    else: error += 1 / len(ious)
    return correct, error, correct/(correct+error)*100

if __name__ == "__main__":
    device = torch.device('cuda')
    image_size = 192
    model = df_basedYolo(image_size, device).cuda()
    model.load_state_dict(torch.load('./checkpoints/201127_212035/df_based_yolo_13_93.tar')['model_state_dict'])
    torch.save(model.state_dict(), './df_based_yolo_weights_192.pt')
    model.eval()

    correct, error, accuracy = evaluate(model, './dog_dataset/eval/dog_images/', './dog_dataset/eval/dog_annotations/', 0.35, 0.5, 0.2, image_size, 64, 0, device, True)

    print(f'accuracy : {accuracy}%')
    print(f'correct : {correct}')
    print(f'error   : {error}')
