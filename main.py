import torch
import torch.utils.data
import torch.utils.tensorboard
from torch.autograd import Variable

import utilData
import util
from model import df_basedYolo
from test import evaluate

import tqdm
import time
import os

if __name__ == '__main__':
    image_size = 192
    device = torch.device('cuda')
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    model = df_basedYolo(image_size, device).cuda()
    util.load_weights(model)
    dataSet = utilData.ListDataset('./dog_dataset/dog_images/', './dog_dataset/dog_annotations/', True, img_size=image_size)
    dataLoader = torch.utils.data.DataLoader(dataSet,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=dataSet.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('logs', now))
    epoch_start = 0
    epoch_num = 1000
    max_accuracy = 0
    max_accuracy_epoch = 0
    for epoch in tqdm.tqdm(range(epoch_start, epoch_start+epoch_num), desc='Epoch'):
        model.train()

        for batch_idx, (_, images, targets) in enumerate(tqdm.tqdm(dataLoader, desc='Batch', leave=False)):
            step = len(dataLoader) * epoch + batch_idx
            # 그래픽카드에 입력
            images = images.to(device)
            targets = targets.to(device)

            # 손실값
            loss, output = model(images, targets)
            optimizer.zero_grad()
            loss.backward()
            # 역전파
            optimizer.step()

            loss_log.set_description_str(f'Loss: {loss.item():.6f}')

            tensorboard_log = []
            for i, yolo_layer in enumerate(model.yolo_layers):
                writer.add_scalar('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
                writer.add_scalar('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
                writer.add_scalar('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
            writer.add_scalar('total_loss', loss.item(), step)

        # lr scheduler의 step을 진행
        scheduler.step()
        correct, error, accuracy = evaluate(model, './dog_dataset/eval/dog_images/', './dog_dataset/eval/dog_annotations/', 0.35, 0.5, 0.2, image_size, 64, 0, device)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_epoch = epoch
        print(f'accuracy : {accuracy}')
        print(f'max accuracy : {max_accuracy}% epoch : {max_accuracy_epoch}')
        writer.add_scalar('eval_accuracy', accuracy, epoch)

        model.train()
        # checkpoint file 저장
        save_dir = os.path.join('checkpoints', now)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(save_dir, f'df_based_yolo_{epoch}_{int(accuracy)}.tar'))
    print(f'max accuracy : {max_accuracy}% epoch : {max_accuracy_epoch}')