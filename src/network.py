from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from utils import compute_same_padding2d
import logging
from math import exp
import numpy as np
from collections import OrderedDict, namedtuple

from torch.nn import init

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        if torch.isnan(grad_output).any():
            return grad_output.zero_()
        else:
            return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, NL='relu',same_padding=True, bn=False, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, dilation=dilation, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, dilation=1, bn=False, bias=True, groups=1):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None

    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class _Conv2d_dilated(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        super(_Conv2d_dilated, self).__init__(
            in_channels, out_channels, kernel_size, stride, _pair(0), dilation,
            False, _pair(0), groups, bias)

    def forward(self, input, dilation=None):
        input_shape = list(input.size())
        dilation_rate = self.dilation if dilation is None else _pair(dilation)
        padding, pad_input = compute_same_padding2d(input_shape, kernel_size=self.kernel_size, strides=self.stride, dilation=dilation_rate)

        if pad_input[0] == 1 or pad_input[1] == 1:
            input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])])
        return F.conv2d(input, self.weight, self.bias, self.stride,
                       (padding[0] // 2, padding[1] // 2), dilation_rate, self.groups)
        #https://github.com/pytorch/pytorch/issues/3867

class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SequentialEndpoints(nn.Module):

    def __init__(self, layers, endpoints=None):
        super(SequentialEndpoints, self).__init__()
        assert isinstance(layers, OrderedDict)
        for key, module in layers.items():
            self.add_module(key, module)
        if endpoints is not None:
            self.Endpoints = namedtuple('Endpoints', endpoints.values(), verbose=True)
            self.endpoints = endpoints


    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def sub_forward(self, startpoint, endpoint):
        def forward(input):
            flag = False
            for key, module in self._modules.items():
                if startpoint == endpoint:
                    output = input
                    if key == startpoint:
                        output = module(output)
                        return output
                elif flag or key == startpoint:
                    if key == startpoint:
                        output = input
                    flag = True
                    output = module(output)
                    if key == endpoint:
                        return output
            return output
        return forward

    def forward(self, input, require_endpoints=False):
        if require_endpoints:
            endpoints = self.Endpoints([None] * len(self.endpoints.keys()))
        for key, module in self._modules.items():
            input = module(input)
            if require_endpoints and key in self.endpoints.keys():
                setattr(endpoints, self.endpoints[key], input)
        if require_endpoints:
            return input, endpoints
        else:
            return input

def save_net(fname, net):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    import h5py
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            if k in h5f.keys():
                del h5f[k]
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net, skip=False, prefix=''):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        for k, v in net.state_dict().items():
            if skip:
                if 'relu' in k:
                    v.copy_(torch.from_numpy(np.zeros((1,))))
                    continue
            if 'loss' in k:
                # print(k)
                continue
            assert (prefix + k) in h5f.keys(), "key: {} size: {}".format(k, v.size())
            param = torch.from_numpy(np.asarray(h5f[(prefix + k)]))
            assert v.size() == param.size(), "{}: h5~{}-need~{}".format(k, param.size(), v.size())
            v.copy_(param)

def diff_net(fname, net):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        for k, v in net.state_dict().items():
            assert k in h5f.keys(), "key: {} size: {}".format(k, v.size())
            param = torch.from_numpy(np.asarray(h5f[k]))
            assert v.size() == param.size(), "{}: h5~{}-need~{}".format(k, param.size(), v.size())
            print("{}: {}".format(k, torch.mean(v - param.cuda())))


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        if '0.3.1' not in torch.__version__ and '0.3.0' not in torch.__version__:  
            with torch.no_grad():
                v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False)
        else:
            v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        # v = v.cuda(non_blocking=True)
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

            elif isinstance(m, nn.LSTM):
                for weight_set in m._all_weights:
                    for param in weight_set:
                        if 'weight' in param:
                            m.__getattr__(param).data.normal_(0.0, dev)
                        if 'bias' in param:
                            m.__getattr__(param).data.fill_(0.0)

            elif isinstance(m, _Conv2d_dilated):
                m.weight.data.copy_(m.weight.data.normal_(0.0, dev))
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight.data, 1.0, 0.02)
                init.constant(m.bias.data, 0.0)