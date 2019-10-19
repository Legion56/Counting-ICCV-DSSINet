import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from utils import compute_same_padding2d
import collections
from network import load_net, _Conv2d_dilated, SequentialEndpoints
import math


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features


    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, output_stride=8, base_dilated_rate=1, NL='relu', bias=True):
    layers = []
    in_channels = 3
    layers = collections.OrderedDict()
    idx = 0
    curr_stride = 1
    dilated_rate = base_dilated_rate
    for v in cfg:
        name, ks, padding = str(idx), (3, 3), (1, 1)
        if type(v) is tuple:
            if len(v) == 2:
                v, ks = v
            elif len(v) == 3:
                name, v, ks = v
            elif len(v) == 4:
                name, v, ks, padding = v
        if v == 'M':
            if curr_stride >= output_stride:
                dilated_rate = 2
                curr_stride *= 2
            else:
                layers[name] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                curr_stride *= 2
            idx += 1
        elif v == 'None':
            idx += 1
        else:
            # conv2d = _Conv2d_dilated(in_channels, v, dilation=dilated_rate, kernel_size=ks, bias=bias)
            conv2d = nn.Conv2d(in_channels, v, dilation=dilated_rate, kernel_size=ks, padding=padding, bias=bias)
            dilated_rate = base_dilated_rate
            layers[name] = conv2d
            idx += 1
            if batch_norm:
                layers[str(idx)] = nn.BatchNorm2d(v)
                idx += 1
            if NL == 'relu' :
                relu = nn.ReLU(inplace=True)
            if NL == 'nrelu' :
                relu = nn.ReLU(inplace=False)
            elif NL == 'prelu':
                relu = nn.PReLU()
            layers['relu'+str(idx)] = relu
            idx += 1
            in_channels = v
    print("\n".join(["{}: {}-{}".format(i, k, v) for i, (k,v) in enumerate(layers.items())]))
    return SequentialEndpoints(layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256],
    'H': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'None', 512, 512, 512, 'None', 512, 512, 512],
    'I': [24, 22, 'M', 41, 51, 'M', 108, 89, 111, 'M', 184, 276, 228],
}

def vgg16(struct='F', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg[struct], **kwargs))

    return model
