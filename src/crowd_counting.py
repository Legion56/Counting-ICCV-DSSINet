from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn import DataParallel
from torch.autograd import Variable
from network import Conv2d, FC, Conv2d_dilated, np_to_variable

import network
import numpy as np

import importlib

class CrowdCounter(nn.Module):

    def __init__(self, optimizer, opt):
        super(CrowdCounter, self).__init__()
        self.opt = opt
        self.device = opt.gpus[0]
        self.model = self.find_model_using_name(opt.model_name)()
        self.loss_fn_ = self.find_loss_using_name(opt.loss or 'MSE')()
        self.init_model(opt.pretrain)

        if optimizer is not None:
            self.optimizer = optimizer(self)
            self.optimizer.zero_grad()

    @property
    def loss(self):
        return self.loss_ 


    def init_model(self, model_path=None):
        if model_path is not None:
            network.load_net(model_path, self.model)
        else:
            network.weights_normal_init(self.model, dev=1e-6)
            # network.load_net('../../pruned_VGG.h5', self.model.front_end, skip=True)
            # network.load_net("../../vgg16.h5", self.model.front_end, skip=True)

        def calpara(model):
            print('---------- Networks initialized -------------')
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
            print('-----------------------------------------------')

        calpara(self.model)

        network.weights_normal_init(self.loss_fn_, dev=0.01)

        if len(self.opt.gpus) > 0:
            assert(torch.cuda.is_available())
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model, self.opt.gpus)  # multi-GPUs
            if self.opt.loss is not None and 'SSIM' in self.opt.loss:
                self.loss_fn_.to(self.device)
                self.loss_fn = torch.nn.DataParallel(self.loss_fn_, self.opt.gpus)  # multi-GPUs
            else:
                self.loss_fn = self.loss_fn_

    
    def forward(self, img_data, gt_data=None, hooker=None, **kargs):
        if self.training:
            img_data = img_data.to(self.device)
            with torch.no_grad():
                noise = img_data.data.new(img_data.shape).uniform_(-0.03,0.03)
                img_data = img_data + noise
            gt_data = gt_data.to(self.device)
        else:
            img_data = img_data.to(self.device)
              


        if self.training:
            density_map = self.model(img_data, **kargs)
        else:
            with torch.no_grad():
                density_map = self.model(img_data, **kargs)

        if self.training:

            self.loss_ = self.loss_fn(density_map, gt_data)
            if len(self.opt.gpus) > 1:
                self.loss_ = self.loss_.mean()

        if hooker is not None:
            return density_map, hooker(self.model.visual)
        else:
            return density_map

    def backward(self, scale=1.0):
        self.loss_ = self.loss_ * scale
        self.loss_.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def find_loss_using_name(self, loss_name):
        # Given the option --model [modelname],
        # the file "models/modelname_model.py"
        # will be imported.
        ssimlib = importlib.import_module('src.ssim')

        if loss_name == 'MSE':
            return nn.MSELoss
        loss_fn = None
        for name, cls in ssimlib.__dict__.items():
            if name.lower() == loss_name.lower():
                print('using loss_fn {}'.format(name))
                loss_fn = cls

        if loss_fn is None:
            print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
            exit(0)

        return loss_fn

    def find_model_using_name(self, model_name):
        # Given the option --model [modelname],
        # the file "models/modelname_model.py"
        # will be imported.
        model_filename = "models." + model_name
        modellib = importlib.import_module(model_filename)

        # In the file, the class called ModelNameModel() will
        # be instantiated. It has to be a subclass of BaseModel,
        # and it is case-insensitive.
        model = None
        for name, cls in modellib.__dict__.items():
            if name.lower() == model_name.lower():
                model = cls

        if model is None:
            print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
            exit(0)

        return model