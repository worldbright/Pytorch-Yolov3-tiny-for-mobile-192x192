import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms
import numpy as np
import os
from PIL import Image

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def getSquareImage(image, pad_value=0):
    _, h, w = image.shape

    if h == w: return image, [0, 0, 0, 0]
    elif h > w: pad = [(h - w) // 2, (h - w) - ((h - w) // 2), 0, 0]
    else: pad = [0, 0, (w - h) // 2, (w - h) - ((w - h) // 2)]
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str, target_path: str, augment: bool, img_size=416):
        self.image_files = [image_path+p for p in os.listdir(image_path)]
        self.target_files = [target_path+p.split('.')[0]+'.txt' for p in os.listdir(image_path)]
        self.max_objects = 100
        self.augment = augment
        self.batch_count = 0
        self.img_size = img_size

    def __getitem__(self, index):
        image_path = self.image_files[index]

        # augmentations
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # 정사각형 패딩
        image = transforms(Image.open(image_path).convert('RGB'))
        image, pad = getSquareImage(image)
        _, h, w = image.shape
        target_path = self.target_files[index]
        targets = torch.zeros((1, 5))
        if os.path.exists(target_path):
            target = np.loadtxt(target_path)

            x1 = target[0] + pad[0]
            y1 = target[1] + pad[2]
            x2 = target[2] + pad[0]
            y2 = target[3] + pad[2]

            targets[:, 1] = ((x1 + x2) / 2) / w
            targets[:, 2] = ((y1 + y2) / 2) / h
            targets[:, 3] = (x2 - x1) / w
            targets[:, 4] = (y2 - y1) / h
        else:
            targets[:, 1] = 0
            targets[:, 2] = 0
            targets[:, 3] = 0
            targets[:, 4] = 0
        # Apply augmentations
        if self.augment and targets is not None:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # Resize images to input shape
        images = torch.stack([F.interpolate(image.unsqueeze(0), self.img_size, mode='bilinear', align_corners=True).squeeze(0) for image in images])
        self.batch_count += 1

        return paths, images, targets