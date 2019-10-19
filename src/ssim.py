from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from utils import compute_same_padding2d
from network import Conv2d_dilated, Conv2d
import logging
from math import exp
import numpy as np
import network
import itertools
import debug

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window / window.sum()


def t_ssim(img1, img2, img11, img22, img12, window, channel, dilation=1, size_average=True):
    window_size = window.size()[2]
    input_shape = list(img1.size())

    padding, pad_input = compute_same_padding2d(input_shape, \
                                                kernel_size=(window_size, window_size), \
                                                strides=(1,1), \
                                                dilation=(dilation, dilation))
    if img11 is None:
        img11 = img1 * img1
    if img22 is None:
        img22 = img2 * img2
    if img12 is None:
        img12 = img1 * img2

    if pad_input[0] == 1 or pad_input[1] == 1:
        img1 = F.pad(img1, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img2 = F.pad(img2, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img11 = F.pad(img11, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img22 = F.pad(img22, [0, int(pad_input[0]), 0, int(pad_input[1])])
        img12 = F.pad(img12, [0, int(pad_input[0]), 0, int(pad_input[1])])

    padd = (padding[0] // 2, padding[1] // 2)

    mu1 = F.conv2d(img1, window , padding=padd, dilation=dilation, groups=channel)
    mu2 = F.conv2d(img2, window , padding=padd, dilation=dilation, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    si11 = F.conv2d(img11, window, padding=padd, dilation=dilation, groups=channel)
    si22 = F.conv2d(img22, window, padding=padd, dilation=dilation, groups=channel)
    si12 = F.conv2d(img12, window, padding=padd, dilation=dilation, groups=channel)

    sigma1_sq = si11 - mu1_sq
    sigma2_sq = si22 - mu2_sq
    sigma12 = si12 - mu1_mu2

    C1 = (0.01*255)**2
    C2 = (0.03*255)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))


    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs

class NORMMSSSIM(torch.nn.Module):

    def __init__(self, sigma=1.0, levels=5, size_average=True, channel=1):
        super(NORMMSSSIM, self).__init__()
        self.sigma = sigma
        self.window_size = 5
        self.levels = levels
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', create_window(self.window_size, self.channel, self.sigma))
        self.register_buffer('weights', torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))


    def forward(self, img1, img2):
        img1 = (img1 + 1e-12) / (img2.max() + 1e-12)
        img2 = (img2 + 1e-12) / (img2.max() + 1e-12)

        img1 = img1 * 255.0
        img2 = img2 * 255.0

        msssim_score = self.msssim(img1, img2)
        return 1 - msssim_score

    def msssim(self, img1, img2):
        levels = self.levels
        mssim = []
        mcs = []

        img1, img2, img11, img22, img12 = img1, img2, None, None, None
        for i in range(levels):
            l, cs = \
                    t_ssim(img1, img2, img11, img22, img12, \
                                Variable(getattr(self, "window"), requires_grad=False),\
                                self.channel, size_average=self.size_average, dilation=(1 + int(i ** 1.5)))

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
            mssim.append(l)
            mcs.append(cs)

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        weights = Variable(self.weights, requires_grad=False)

        return torch.prod(mssim ** weights)