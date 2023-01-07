import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255                       
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class Custom_Dataset(Dataset):
    def __init__(self, root, transform, csv_path):
        super().__init__()
        self.root = root
        self.transform = transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']
        file_path = glob(self.root+'/*/'+file_name)[0]
        file_name = file_name.split('.')[0]
        label = patience_info['label']
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)

        return {'imgs': img, 'labels': label, 'names': file_name}

    def __len__(self):
        return len(self.info)

def get_dataset(imgpath, csvpath, img_size, mode='train', keyword=None):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        # transforms.CenterCrop((img_size, img_size)),
        # AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        # AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
            transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = Custom_Dataset(imgpath, transform, csvpath)

    return dataset

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix