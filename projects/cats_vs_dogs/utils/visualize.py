# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 23:51:27 2019

@author: Snake
"""


import visdom
import time
import numpy as np


class Visualizer(object):
    # 封装 visdom 的基本操作，但仍然可以通过self.vis.function调用原生 visdom 接口
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''
        
    def reinit(self, env='default', **kwargs):
        # 修改 visdom 配置
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self
    
    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)
            
    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)
            
    def plot(self, name, y,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        self.vis.line(Y=np.array(y), X=np.arange(len(y)),
                      win=str(name),
                      opts=dict(title=name),
                      update=None,
                      **kwargs
                      )

    def img(self, name, img_,**kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                       win=str(name),
                       opts=dict(title=name),
                       **kwargs
                       )


    def log(self,dic,win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        self.log_text = ''
        if 'epoch' in dic.keys():
            
            for i in range(len(dic['epoch'])):
                info = "{name},epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                            name=dic['name'][i],epoch=dic['epoch'][i],loss=dic['loss'][i],lr=dic['lr'][i],
                            val_cm=str(dic['val_cm'][i]),train_cm=str(dic['train_cm'][i]))
                '''
                self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info))
                '''
                self.log_text += ('{info} <br>'.format(info=info))
        self.vis.text(self.log_text,win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
