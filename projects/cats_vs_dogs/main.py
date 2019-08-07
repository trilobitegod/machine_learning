# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 00:11:23 2019

@author: Snake
"""


import os
import torch
#torch.cuda.current_device()
#torch.cuda._initialized = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from tqdm import tqdm

import models
from data import DogCat
from config import opt
from utils.visualize import Visualizer
from config import load_dict
from config import save_dict

def test(**kwargs):
    opt.parse(kwargs)
    import ipdb;
    ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    
    if os.path.exists(opt.load_model_path):
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = Variable(data, volatile=True)
        if opt.user_gpu: input = input.cuda()
        score = model(input)
        probability = F.softmax(score)[:, 0].data.tolist()
        
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        
        results += batch_results
    write_csv(results, opt.results_file)
    
def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)
        
def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    
    # step1: configure model
    model = getattr(models, opt.model)()
    if os.path.exists(opt.load_model_path):
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
   
    if os.path.exists(opt.pars_path):
        dic = load_dict(opt.pars_path)
        previous_loss = dic['loss'][-1] if 'loss' in dic.keys() else 1e100
    else:
        dic ={}
    # step2: data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    
    # step2: criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    #previous_loss = 1e100
    
    # train
    for epoch in range(3, opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        '''
        for ii, (data, label) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            #confusion_matrix.reset()
            # train model
            input = Variable(data)
            target = Variable(label)                
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
                
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            
            # meters update and visualize
            loss_meter.add(loss.data.item())
            confusion_matrix.add(score.data, target.data)
            if ii % opt.print_freq == opt.print_freq - 1:
                dic = save_dict(opt.pars_path, dic, loss_data=loss_meter.value()[0])
                #loss_meter.reset()
                vis.plot('loss', dic['loss_data'])
                model.save()
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trave()
            if ii==200: break


        # update learning: reduce learning rate when loss no longer decrease
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        dic = save_dict(opt.pars_path, dic, epoch=epoch, lr=lr, loss=loss_meter.value()[0], 
                      train_cm=confusion_matrix.value())
	'''
        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)        
        dic = save_dict(opt.pars_path, dic, val_accuracy=val_accuracy, val_cm=val_cm.value())
        
        vis.log(dic)
        vis.plot('val_accuracy', dic['val_accuracy'])

                
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix1 = meter.ConfusionMeter(2)
    confusion_matrix1.reset()
    for ii, (data, label) in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            val_input = Variable(data)
            val_label = Variable(label)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix1.add(score.data, val_label.data)
        
    model.train()
    cm_value = confusion_matrix1.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix1, accuracy

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)
    
if __name__=='__main__':
    import fire
    fire.Fire()
    train()
