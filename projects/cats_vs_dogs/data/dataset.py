# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:27:52 2019

@author: Snake
"""


import os
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as T


class DogCat(data.Dataset):
    
    def __init__(self, root, transforms=None, train=True, test=False):
        # 获取图片地址，并根据训练、验证、测试划分数据
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
            
        imgs_num = len(imgs)
        
        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)
        
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.9*imgs_num):]
            
        if transforms is None:
            normalize = T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
            
            if self.test or not train:
                self.transforms = T.Compose([
                        T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        normalize
                        ])
            else:
                self.transforms = T.Compose([
                        T.Resize(256),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize
                        ])
    
    def __getitem__(self, index):
        # 一次返回一张图片数据
        img_path = self.imgs[index]
        if self.test: 
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return len(self.imgs)