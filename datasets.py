import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class KaistPDDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        # fusion을 진행할 것이기 때문에 rgb와 thermal 이미지를 모두 불러옵니다
        with open(os.path.join(data_folder, self.split + '_rgb_images.json'), 'r') as j:
            self.rgb_images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_lwir_images.json'), 'r') as j:
            self.thermal_images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.rgb_images) == len(self.thermal_images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # rgb와 thermal 모두 open, convert 합니다.
        rgb_image = Image.open(self.rgb_images[i], mode='r')
        rgb_image = rgb_image.convert('RGB')
        
        thermal_image = Image.open(self.thermal_images[i], mode='r')
        thermal_image = thermal_image.convert('RGB')
        
        # object json은 rgb와 thermal 동일하게 때문에 변경하지 않앗습니다.
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['category_id'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        # rgb, thermal 이미지 변형한 것을 받아줍니다.
        rgb_image, boxes, labels, difficulties = transform(rgb_image, boxes, labels, difficulties, split=self.split)
        thermal_image, boxes, labels, difficulties = transform(thermal_image, boxes, labels, difficulties, split=self.split)

        image_list = []
        image_list.append(rgb_image)
        image_list.append(thermal_image)
        
        # image_list로 rgb_image와 thermal_image 모두를 return해줍니다
        # image_list[0] == rgb_image
        # image_list[1] == thermal_image
        return image_list, boxes, labels, difficulties

    def __len__(self):
        return len(self.rgb_images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        
        # image 리스트로 이미지가 들어오기 때문에 수정이 필요합니다
        # shape 주의!!
        rgb_images = list()
        thermal_images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            # len(batch) # 32
            # len(batch[0]) # 4
            # len(batch[0][0]) # 2
            # len(batch[0][0][0]) # 3 (이미지 접근)
            rgb_images.append(b[0][0])
            thermal_images.append(b[0][1])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        
        rgb_images = torch.stack(rgb_images, dim=0)
        thermal_images = torch.stack(thermal_images, dim=0)

        image_list = []
        image_list.append(rgb_images)
        image_list.append(thermal_images)

        return image_list, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
