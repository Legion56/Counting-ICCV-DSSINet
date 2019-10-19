import numpy as np
from scipy.ndimage import imread
import os
import random
import sys
import itertools
from density_gen import Gauss2D, read_image_label_fix, read_image_label_apdaptive, \
read_image_label_3d, read_image, save_density_map, get_annoted_kneighbors

import copy
import re

from collections import namedtuple
from src.timer import Timer

basic_config = {
    'fixed': {
        "sigma": 4.0, "f_sz": 15.0, "channels": 3, "downsize": 32
    },
    'adaptive': {
        "K": 4, "channels": 3, "downsize": 32
    },
    '3d': {
        "K":4, "S": [9,25,49,81], "channels": 3, "downsize": 32
    },
    'unlabel': {
        "channels": 3, "downsize": 32
    }
}

mode_func = {
    'fixed': read_image_label_fix,
    'adaptive': read_image_label_apdaptive,
    '3d': read_image_label_3d,
    'unlabel': read_image
}

Blob = namedtuple('Blob', ('img', 'den', 'gt_count'))

class ImageDataLoader():
    def __init__(self, image_path, label_path, mode, is_preload=True, split=None, annReadFunc=None, **kwargs):

        self.image_path = image_path
        self.label_path = label_path

        self.image_files = [filename for filename in os.listdir(image_path) \
                           if os.path.isfile(os.path.join(image_path,filename))]
        self.label_files = [filename for filename in os.listdir(label_path) \
                           if os.path.isfile(os.path.join(label_path,filename))]


        self.image_files.sort(cmp=lambda x, y: cmp('_'.join(re.findall(r'\d+',x)),'_'.join(re.findall(r'\d+',y))))
        self.label_files.sort(cmp=lambda x, y: cmp('_'.join(re.findall(r'\d+',x)),'_'.join(re.findall(r'\d+',y))))

        for img, lab in zip(self.image_files, self.label_files):
            assert '_'.join(re.findall(r'\d+', img)) == '_'.join(re.findall(r'\d+',lab))

        if split != None:
            self.image_files = split(self.image_files)
            self.label_files = split(self.label_files)
        self.num_samples = len(self.image_files)
        self.mode = mode
        self.annReadFunc = annReadFunc

        self.blob_list = []

        self.fspecial = Gauss2D()
        self.is_preload = is_preload
        self.read_func_kwargs = kwargs

        if 'test' in kwargs.keys():
            self.test = kwargs['test']
        else:
            self.test = False

        if self.mode == 'adaptive':
            self.precompute_scale()
            print("K neighbors for adaptive density map Done.")
        if self.is_preload:
            self.preload_data()
        

    def preload_data(self):
        print 'Pre-loading the data. This may take a while...'

        t = Timer()
        t.tic()
        self.blob_list = [_ for _ in range(self.num_samples)]
        self.is_preload = False
        for i in range(self.num_samples):
            self.blob_list[i] = (self.load_index(i))
            if i % 50 == 0:
                print "loaded {}/{} samples".format(i, self.num_samples)
        duration = t.toc(average=False)
        print 'Completed loading ' ,len(self.blob_list), ' files, time: ', duration
        self.is_preload = True

    def precompute_scale(self):
        self.kneighbors = []
        for i in range(self.num_samples):
            neighbors = get_annoted_kneighbors(self.label_files[i], self.label_path, \
                            K=self.read_func_kwargs['K'], annReadFunc=self.annReadFunc)
            self.kneighbors += [neighbors]

    def load_index(self, i):
        image_file, label_file = self.image_files[i], self.label_files[i]
        if self.mode != 'adaptive':
            img, den, gt_count = mode_func[self.mode](image_file, label_file, self.image_path, self.label_path, \
                                        self.fspecial.get, annReadFunc=self.annReadFunc, **self.read_func_kwargs)
        else:
            img, den, gt_count = mode_func[self.mode](image_file, label_file, self.image_path, self.label_path, \
                                        self.fspecial.get, annReadFunc=self.annReadFunc, kneighbors=self.kneighbors[i], \
                                        **self.read_func_kwargs)

        return Blob(img, den, gt_count)

    def query_fname(self, i):
        return self.image_files[i]

    def __getitem__(self, i):
        return self.__index__(i)

    def __index__(self, i):
        if self.is_preload:
            blob = self.blob_list[i]
            return blob
        else:
            return self.load_index(i)
    def __iter__(self):
        for i in range(self.num_samples):
            yield self.__index__(i)

    def get_num_samples(self):
        return self.num_samples

    def __len__(self):
        return self.num_samples

