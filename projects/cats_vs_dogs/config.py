# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:00:11 2019

@author: Snake
"""


import warnings
import json
import numpy as np
import datetime
class DefaultConfig(object):
    env = 'default'
    model = 'ResNet34'
    
    train_data_root = './data/train/'
    test_data_root = './data/test1/'
    load_model_path = './checkpoints/resnet34_20190807_123952.pth'
    pars_path = './checkpoints/pars.json'
    batch_size = 32
    use_gpu = False
    num_workers = 0 # how many workers for loading data
    print_freq = 20 # print info for energy N bath
    
    debug_file = './tmp/debug'
    result_file = 'result.csv'
    
    max_epoch = 10
    lr = 0.1 * 0.95
    lr_decay = 0.95
    weight_decay = 1e-4
    
    
def parse(self, kwargs):
    # 根据字典 kwargs 更新 config 参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn('Warning: opt has not attribute %s' % k)
        setattr(self, k, v)
        
    print('user config:')
    
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):                                 
            return obj.__str__()
        else:
            return super(JsonEncoder, self).default(obj)

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
	    dic = json.load(json_file)
    return dic

def save_dict(filename, dic, **kwargs):
    '''save dict into json file'''
    for k, v in kwargs.items():
        if k not in dic.keys():
            dic[k] = []
        dic[k].append(v)
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)
    return dic
            
DefaultConfig.parse = parse
opt = DefaultConfig()


