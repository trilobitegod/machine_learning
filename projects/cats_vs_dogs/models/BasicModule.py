# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:28:53 2019

@author: Snake
"""


import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    # 封装nn.Module，主要提供 sace 和 load 方法
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))   # 默认名字
        
    def load(self, path):
        # 可加载指定路径的模型
        self.load_state_dict(torch.load(path))
        
    def save(self, name=None):
        # 保存模型，默认使用 “ 模型名字+时间 ” 作为文件名
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%Y%m%d_%H%M%S.pth')
        torch.save(self.state_dict(), name)
        return name
    
    
class Flat(nn.Module):
    # 把输入reshape成 (batch_size, dim_length)
    def __init__(self):
        super(Flat, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)