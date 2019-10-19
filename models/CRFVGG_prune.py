#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.network import Conv2d, FC, Conv2d_dilated, np_to_variable
from src.vgg import vgg16
import numpy as np
from torch.autograd import Variable
import logging


class MessagePassing(nn.Module):
    def __init__(self, branch_n, input_ncs, bn=False):
        super(MessagePassing, self).__init__()
        self.branch_n = branch_n
        self.iters = 2
        for i in range(branch_n):
            for j in range(branch_n):
                if i == j:
                    continue
                setattr(self, "w_0_{}_{}_0".format(j, i), \
                        nn.Sequential(
                                Conv2d_dilated(input_ncs[j],  input_ncs[i], 1, dilation=1, same_padding=True, NL=None, bn=bn),
                            )
                        )
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU()
        
    def forward(self, input):
        hidden_state = input
        side_state = []

        for _ in range(self.iters):
            hidden_state_new = []
            for i in range(self.branch_n):

                unary = hidden_state[i]
                binary = None
                for j in range(self.branch_n):
                    if i == j:
                        continue
                    if binary is None:
                        binary = getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])
                    else:
                        binary = binary + getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])

                binary = self.prelu(binary)
                hidden_state_new += [self.relu(unary + binary)]
            hidden_state = hidden_state_new

        return hidden_state

class CRFVGG_prune(nn.Module):
    def __init__(self, output_stride=8, bn=False):
        super(CRFVGG_prune, self).__init__()

        self.output_stride = output_stride

        self.pyramid = [2, 0.5]

        self.front_end = vgg16(struct='I', NL="prelu", output_stride=self.output_stride)



        # [24, 22, 'M', 41, 51, 'M', 108, 89, 111, 'M', 184, 276, 228],
        self.passing1 = MessagePassing( branch_n=2, 
                                        input_ncs=[51, 22],
                                        )
        self.passing2 = MessagePassing( branch_n=3, 
                                        input_ncs=[111, 51, 22],
                                        )
        self.passing3 = MessagePassing( branch_n=3, 
                                        input_ncs=[228, 111, 51],
                                        )
        self.passing4 = MessagePassing( branch_n=2, 
                                        input_ncs=[228, 111],
                                        )

        self.decoder1 = nn.Sequential(
                Conv2d_dilated(228, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )
        self.decoder2 = nn.Sequential(
                Conv2d_dilated(339, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )
        self.decoder3 = nn.Sequential(
                Conv2d_dilated(390, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder4 = nn.Sequential(
                Conv2d_dilated(184, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder5 = nn.Sequential(
                Conv2d_dilated(73, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.passing_weight1 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight2 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight3 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight4 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()

    def forward(self, im_data, return_feature=False):
        conv1_2 = ['0', 'relu3']
        conv1_2_na = ['0', '2']
        conv2_2 = ['4', 'relu8']
        conv2_2_na = ['4', '7']
        conv3_3 = ['9', 'relu15']
        conv3_3_na = ['9', '14']
        # layer 16 is the max pooling layer
        conv4_3 = ['16', 'relu22']
        conv4_3_na = ['16', '21']

        # droping the last pooling layer, 17 would become dilated with rate 2
        # conv4_3 = ['17', 'relu22']

        batch_size, C, H, W = im_data.shape

        with torch.no_grad():
            im_scale1 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[0]), int(W * self.pyramid[0])), align_corners=False, mode="bilinear")
            im_scale2 = im_data
            im_scale3 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[1]), int(W * self.pyramid[1])), align_corners=False, mode="bilinear")


        mp_scale1_feature_conv2_na = self.front_end.features.sub_forward(conv1_2[0], conv2_2_na[1])(im_scale1)
        mp_scale2_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale2)

        mp_scale1_feature_conv2, mp_scale2_feature_conv1 \
                        = self.passing1([mp_scale1_feature_conv2_na, mp_scale2_feature_conv1_na])


        aggregation4 = torch.cat([mp_scale1_feature_conv2, mp_scale2_feature_conv1], dim=1)

        mp_scale1_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale1_feature_conv2)
        mp_scale2_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale2_feature_conv1)
        mp_scale3_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale3)


        mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1 \
                        = self.passing2([mp_scale1_feature_conv3_na, mp_scale2_feature_conv2_na, mp_scale3_feature_conv1_na])
        aggregation3 = torch.cat([mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1], dim=1)


        mp_scale1_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale1_feature_conv3)
        mp_scale2_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale2_feature_conv2)
        mp_scale3_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale3_feature_conv1)

        mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2 \
                        = self.passing3([mp_scale1_feature_conv4_na, mp_scale2_feature_conv3_na, mp_scale3_feature_conv2_na])
        aggregation2 = torch.cat([mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2], dim=1)

        mp_scale2_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale2_feature_conv3)
        mp_scale3_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale3_feature_conv2)

        mp_scale2_feature_conv4, mp_scale3_feature_conv3 \
                        = self.passing4([mp_scale2_feature_conv4_na, mp_scale3_feature_conv3_na])
        aggregation1 = torch.cat([mp_scale2_feature_conv4, mp_scale3_feature_conv3], dim=1)

        mp_scale3_feature_conv4 = self.front_end.features.sub_forward(*conv4_3)(mp_scale3_feature_conv3)

        dens1 = self.decoder1(mp_scale3_feature_conv4)
        dens2 = self.decoder2(aggregation1)
        dens3 = self.decoder3(aggregation2)
        dens4 = self.decoder4(aggregation3)
        dens5 = self.decoder5(aggregation4)

        dens1 = self.prelu(dens1)
        dens2 = self.prelu(dens2 + self.passing_weight1(nn.functional.upsample(dens1, scale_factor=2, align_corners=False, mode="bilinear")))
        dens3 = self.prelu(dens3 + self.passing_weight2(nn.functional.upsample(dens2, scale_factor=2, align_corners=False, mode="bilinear")))
        dens4 = self.prelu(dens4 + self.passing_weight3(nn.functional.upsample(dens3, scale_factor=2, align_corners=False, mode="bilinear")))
        dens5 = self.relu(dens5 + self.passing_weight4(nn.functional.upsample(dens4, scale_factor=2, align_corners=False, mode="bilinear")))

        return dens5
