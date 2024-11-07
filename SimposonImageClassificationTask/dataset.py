# -*- coding: utf-8 -*-
import sys
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as tffunc

def get_simpsons_subsets(dataset_path):
    training_set = []
    validation_set = []
    training_labels = []
    validation_labels = []
    class_label_mapping = []
    subfolders = glob.glob(dataset_path + '/*/')
    for label, subfolder in enumerate(subfolders, start=0):
        class_name = subfolder.rstrip('/').split('/')[-1]
        class_label_mapping.append(class_name)
        images = glob.glob(subfolder + '/*.jpg')
        images.sort()
        split_point = int(len(images) * 0.60)
        training_images = images[:split_point]
        validation_images = images[split_point:]
        training_set += training_images
        validation_set += validation_images
        training_labels += [label] * len(training_images)
        validation_labels += [label] * len(validation_images)

    return training_set, training_labels, validation_set, validation_labels, class_label_mapping


class SimpsonsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, class_names, is_validation):
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.is_validation = is_validation

        self.transform_basic = transforms.Compose([
            PadToSquared(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_with_augmentation = transforms.Compose([
            PadToSquared(),
            transforms.Resize((128, 128)),
            # geo augment
            #colorjitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index])

        if self.is_validation:
            transform = self.transform_basic
        else:
            transform = self.transform_with_augmentation

        image = transform(image)

        label = np.asarray(self.labels[index])
        label = torch.from_numpy(label.copy()).long()
        return image, label

    def __len__(self):
        return len(self.images)


def get_dataloader(dataset_path):
    num_workers = 16
    training_set, training_labels, validation_set, validation_labels, class_label_mapping = get_simpsons_subsets(dataset_path)
    train_dataset = SimpsonsDataset(training_set, training_labels, class_label_mapping, is_validation=False)
    val_dataset = SimpsonsDataset(validation_set, validation_labels, class_label_mapping, is_validation=True)
    training_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, drop_last=True)
    validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, drop_last=False)
    return training_loader, validation_loader, class_label_mapping


class PadToSquared:
    def __call__(self, image):
        width, height = image.size
        if width > height:
            padding = (0, width - height, 0, 0)
        else:
            padding = (height - width, 0, 0 , 0)
        return tffunc.pad(image, padding, fill=0, padding_mode='constant')