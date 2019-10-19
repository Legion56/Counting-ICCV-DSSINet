import numpy as np
import cv2
from scipy.ndimage import imread
import os
import random
import sys
import itertools
from timer import Timer
from density_gen import load_annPoints
import copy
from torch.utils.data import DataLoader
import torch
import concurrent.futures
import network
import torchvision.transforms as transforms
import re
from PIL import Image as Image
from PIL import ImageOps as ImageOps

try:
    from timer import Timer
except ImportError:
    from src.timer import Timer


basic_config = {
    'Fixed': {
        "crop_size": (224,224),
        "crop_scale_limit": 1e9,
    },
    'Prior': {
        "crop_size": (224,224),
        "crop_scale_limit": 8,
    },
    'Adap': {
        "fixed_size": 256
    }
}

def i_crop(img, pos, size):
    ow, oh = img.size
    pH, pW = pos
    cH, cW = size
    cpH, cpW = oh - cH, ow - cW
    pH, pW = int(pH*cpH), int(pW*cpW)
    return img.crop((pW, pH, pW + cW, pH + cH))

def i_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def multi_crop(img, crops, cH, cW):
    return [img.crop((x, y, x + cW, y + cH)) for (x, y) in crops]

def d_crop(den, pos, size):
    oh, ow = den.shape
    pH, pW = pos
    cH, cW = size
    cpH, cpW = oh - cH, ow - cW
    pH, pW = int(pH*cpH), int(pW*cpW)
    return den[np.newaxis, pH:pH + cH, pW:pW + cW].copy()

def d_flip(den, flip):
    if flip:
        return np.flip(den,2).copy()
    return den

class FixedSampler():
    def __init__(self, dataloader, fixed_crop=False, crop_size=224, crop_scale_limit=8, \
                 patches_per_sample=5, shuffle=False):
        self.dataloader = dataloader
        self.num_samples = len(dataloader)
        self.patches_per_sample = patches_per_sample
        self.crop_size = crop_size
        self.crop_scale_limit = crop_scale_limit

        self.shuffle = shuffle

        self.patches = []

        if shuffle:
            random.seed(2468)

        self.patch_generate(self.num_samples, self.patches_per_sample)

        self.patch_list = list(itertools.chain(
                                *[[(i,j) for j in range(len(patch))] for i, patch in enumerate(self.patches)]))

        self.fixed_crop = fixed_crop

        self.training = not self.dataloader.test

        if shuffle:
            self.shuffle_list()


    def patch_generate(self, num_samples, num_patches):
        cor = [[(random.random(), random.random()) for _ in range(num_patches)] for _ in range(num_samples)]
        def slicer(i):
            p = [(0,0),(0,1),(1,0),(1,1)]
            for c in cor[i]:
                p += [(c[0], c[1])]
            return p#random.choice(p)
        self.patches = [slicer(i) for i in range(self.num_samples)]
        print "recroping"

    def query_fname(self, i):
        return self.dataloader.query_fname(self.patch_list[i][0])

    def shuffle_list(self):
        if not self.fixed_crop:
            self.patch_generate(self.num_samples, self.patches_per_sample)
            self.patch_list = list(itertools.chain(
                                    *[[(i,j) for j in range(len(patch))] for i, patch in enumerate(self.patches)]))
            self.filps = [random.uniform(0.0, 1.0) > 0.5 for _ in range(len(self.patch_list))]

        random.shuffle(self.patch_list)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_list()
            raw_input(len(self.filps))
            
        for i in range(len(self.patch_list)):
            yield self.__index__(i)

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, i):
        '''bid: image index; pid: patch index, to find slice'''
        bid, pid = self.patch_list[i]
        transform_img = []
        transform_den = []
        transform_raw = []

        transform_raw.append(transforms.Lambda(lambda img: i_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_raw.append(transforms.Lambda(lambda img: i_flip(img, self.filps[i])))
        transform_raw.append(transforms.Lambda(lambda img: np.array(img)))
        transform_raw = transforms.Compose(transform_raw)

        transform_img.append(transforms.Lambda(lambda img: i_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_img.append(transforms.Lambda(lambda img: i_flip(img, self.filps[i])))
        transform_img += [ transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                           ]
        transform_img = transforms.Compose(transform_img)

        transform_den.append(transforms.Lambda(lambda img: d_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_den.append(transforms.Lambda(lambda img: d_flip(img, self.filps[i])))
        transform_den += [transforms.Lambda(lambda den: network.np_to_variable(den, is_cuda=False, is_training=self.training))]
        transform_den = transforms.Compose(transform_den)

        img, den, gt_count = self.dataloader[bid]

        return transform_img(img.copy()), transform_den(den), transform_raw(img.copy()), gt_count, i

    def get_num_samples(self):
        return len(self.patch_list)

    def __len__(self):
        return self.get_num_samples()


# For UCF_QNRF dataset training
class PriorFixedSampler():
    def __init__(self, dataloader, crop_size=224, crop_scale_limit=8, \
                 patches_per_sample=5, shuffle=False, fixed_crop=False, ):
        self.dataloader = dataloader
        self.patches_per_sample = patches_per_sample
        self.crop_size = crop_size#(crop_size, crop_size)
        self.crop_scale_limit = crop_scale_limit

        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)

        self.label_files = dataloader.label_files
        self.label_path = dataloader.label_path
        self.image_files = dataloader.image_files
        self.image_path = dataloader.image_path
        self.annReadFunc = dataloader.annReadFunc

        self.sample_classify()

        self.num_samples = len(self.sample_dict.keys())
        self.patch_generate(self.num_samples, self.patches_per_sample)

        self.patch_list = list(itertools.chain(
                                *[[(i,j) for j in range(len(patch))] for i, patch in enumerate(self.patches)]))

        self.fixed_crop = fixed_crop

        self.training = not self.dataloader.test

        if shuffle:
            self.shuffle_list()


    def patch_generate(self, num_samples, num_patches):
        cor = [[(random.random(), random.random()) for _ in range(num_patches)] for _ in range(num_samples)]
        def slicer(i):
            p = [(0,0),(0,1),(1,0),(1,1)]
            for c in cor[i]:
                p += [(c[0], c[1])]
            return p
        self.patches = [slicer(i) for i in range(self.num_samples)]

    def query_fname(self, i):
        return self.dataloader.query_fname(self.patch_list[i][0])

    def sample_classify(self):
        self.sample_dict = {}
        self.pro_dict = {}

        args = ((self.label_path + '/' + label, self.annReadFunc) for label in self.label_files)
        gt_counts = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for gt_count in executor.map(lambda p: len(load_annPoints(*p)), args):
                gt_counts += [gt_count]
        print "load done"
        rate = [0,0]
        for i, (A_path, B_path) in enumerate(zip(self.image_files, self.label_files)):
            img_id, patch_id = re.findall(r'\d+', B_path)
            gt_count = gt_counts[i]
            # gt_count = len(load_annPoints(self.label_path + '/' + B_path, annReadFunc=self.annReadFunc)) * 1.0
            # assert gt_count == gt_count2
            if int(img_id) not in self.sample_dict.keys():
                self.sample_dict[int(img_id)] = [[],[]]
            if gt_count > 40:
                self.sample_dict[int(img_id)][0] += [(i, self.image_path + '/' + A_path, self.label_path + '/' + B_path)]
                rate[0] += 1
            elif gt_count > 2:
                self.sample_dict[int(img_id)][1] += [(i, self.image_path + '/' + A_path, self.label_path + '/' + B_path)]
                rate[1] += 1
            # else:
            # background patch never chosen
        for i in self.sample_dict.keys():
            pro = np.array([1, 0.5]) * np.array([len(self.sample_dict[i][0]) > 0,\
                                                   len(self.sample_dict[i][1]) > 0])
            pro /= pro.sum()
            self.pro_dict[i] = pro
        print 'class rate: ', rate
 
    def shuffle_list(self):
        if not self.fixed_crop:
            self.patch_generate(self.num_samples, self.patches_per_sample)

            self.patch_list = list(itertools.chain(
                                    *[[(i,j) for j in range(len(patch))] for i, patch in enumerate(self.patches)]))
            self.filps = [random.uniform(0.0, 1.0) > 0.5 for _ in range(len(self.patch_list))]
        random.shuffle(self.patch_list)

    def preload(self):

        self.ncls = {k: np.random.choice([0,1], p=pro) for k, pro in self.pro_dict.items()}

        self.choices = {k: random.randint(0, len(self.sample_dict[k][ncl]) - 1) for k, ncl in self.ncls.items()}
    
        self.loaded = {}
        indexs = [self.sample_dict[self.patch_list[i][0] + 1]\
                                    [self.ncls[self.patch_list[i][0] + 1]]\
                                    [self.choices[self.patch_list[i][0] + 1]]\
                                    [0] for i in range(self.get_num_samples())]
        indexs = set(indexs)
        load_timer = Timer()
        load_timer.tic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for blob in executor.map(lambda i: (i,self.dataloader[i]), indexs):
                self.loaded[blob[0]] = blob[1]
        print "re-prior crop: %f s" % load_timer.toc(average=False)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_list()
        for i in range(len(self.patch_list)):
            yield self.__index__(i)

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, i): 
        bid, pid = self.patch_list[i]

        ncl = self.ncls[bid + 1]
        choice = self.choices[bid  + 1]
        idx, image, label = self.sample_dict[bid + 1][ncl][choice]
        img, den, gt_count = self.loaded[idx]


        transform_img = []
        transform_den = []
        transform_raw = []

        transform_raw.append(transforms.Lambda(lambda img: i_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_raw.append(transforms.Lambda(lambda img: i_flip(img, self.filps[i])))
        transform_raw.append(transforms.Lambda(lambda img: np.array(img)))
        transform_raw = transforms.Compose(transform_raw)

        transform_img.append(transforms.Lambda(lambda img: i_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_img.append(transforms.Lambda(lambda img: i_flip(img, self.filps[i])))
        transform_img += [ transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                           ]
        transform_img = transforms.Compose(transform_img)

        transform_den.append(transforms.Lambda(lambda img: d_crop(img, self.patches[bid][pid], self.crop_size)))
        transform_den.append(transforms.Lambda(lambda img: d_flip(img, self.filps[i])))
        transform_den += [transforms.Lambda(lambda den: network.np_to_variable(den, is_cuda=False, is_training=self.training))]
        transform_den = transforms.Compose(transform_den)


        return transform_img(img.copy()), transform_den(den), transform_raw(img.copy()), gt_count, i

    def get_num_samples(self):
        return len(self.patch_list)

    def __len__(self):
        return self.get_num_samples()


class AdapSlicer():
    def __init__(self, dataloader, fixed_size=256, shuffle=False, **kwargs):
        self.dataloader = dataloader
        self.num_samples = len(self.dataloader)

        self.shuffle = shuffle
        assert self.shuffle == False

        self.fixed_size = fixed_size
        self.blob_list = self.dataloader


    def shuffle_list(self):
        assert False

    def __iter__(self):
        for i in range(self.num_samples):
            yield self.__index__(i)

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, index):
        data, gt_density, gt_count = self.blob_list[index]
        fname = self.dataloader.query_fname(index)
        W, H = data.size
        fixed_size = self.fixed_size
        transform_img = []

        if fixed_size != -1 and not (H % fixed_size == 0 and W % fixed_size == 0):
            pad_h = ((H / fixed_size + 1) * fixed_size - H) % fixed_size
            pad_w = ((W / fixed_size + 1) * fixed_size - W) % fixed_size
            image_pads = (pad_w / 2, pad_h / 2, pad_w - pad_w / 2, pad_h - pad_h / 2)

            transform_img.append(transforms.Pad(image_pads, fill=0))
            H = H + pad_h
            W = W + pad_w
            mask = torch.zeros((H, W),dtype=torch.uint8).byte()
            mask[pad_h / 2:H - (pad_h - pad_h / 2), pad_w / 2:W - (pad_w - pad_w / 2)] = 1
        elif H % fixed_size == 0 and W % fixed_size == 0:
            mask = torch.ones((H, W),dtype=torch.uint8).byte()
        else:
            mask = None 

        normalizor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        if fixed_size != -1:
            crop_indexs = [(x * fixed_size, y * fixed_size) for x, y in itertools.product(range(W / fixed_size), range(H / fixed_size))]
            transform_img.append(transforms.Lambda(lambda img: multi_crop(img, crop_indexs, fixed_size, fixed_size)))
            transform_img.append(transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]))
            transform_img.append(transforms.Lambda(lambda crops: torch.stack([normalizor(crop) for crop in crops])))
        else:
            transform_img.append(transforms.ToTensor())
            transform_img.append(normalizor)

        if self.dataloader.test:
            return index, fname, transforms.Compose(transform_img)(data.copy()), mask, gt_count
        else:
            return index, fname, transforms.Compose(transform_img)(data.copy()), mask, gt_density, gt_count


    def get_loader(self, batch_size=1):
        return DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=12)

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return self.num_samples


mode_func = {
    'Fixed': FixedSampler,
    'Prior': PriorFixedSampler,
    'Adap': AdapSlicer
}


if __name__ == '__main__':
    pass