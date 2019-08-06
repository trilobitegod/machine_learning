# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:31:46 2019

@author: Snake
"""


# 卷积层输出维度

def dim_conv2d(input_size, kernel_size, stride=1, padding=0):
    return int((input_size + 2 * padding -kernel_size) / stride) + 1



out = dim_conv2d(dim_conv2d(256,11,1,4),11,1,4)

print(out)