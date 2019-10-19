import cv2
import scipy.io as scio
from scipy.ndimage import imread
from scipy.misc import imsave
import os
import re
import numpy as np
import itertools

def crop():
    image_path = '/UCF-QNRF_ECCV18/Train/images'
    label_path = '/UCF-QNRF_ECCV18/Train/ground_truth'
    image_save_path = '/UCF-QNRF_ECCV18/Train/crop_images1'
    label_save_path = '//UCF-QNRF_ECCV18/Train/crop_ground_truth1'
    crop_size = 512

    image_files = [filename for filename in os.listdir(image_path) \
                       if os.path.isfile(os.path.join(image_path,filename))]
    label_files = [filename for filename in os.listdir(label_path) \
                       if os.path.isfile(os.path.join(label_path,filename))]
    image_files.sort(cmp=lambda x, y: cmp((re.findall(r'\d+',x)[0]),(re.findall(r'\d+',y)[0])))
    label_files.sort(cmp=lambda x, y: cmp((re.findall(r'\d+',x)[0]),(re.findall(r'\d+',y)[0])))

    # print image_files
    # print label_files

    for image_file, label_file in zip(image_files, label_files):

        img = imread(os.path.join(image_path, image_file), 0)
        img = img.astype(np.float32, copy=False)
        hh = img.shape[0]
        ww = img.shape[1]
        annPoints = scio.loadmat((os.path.join(label_path, label_file)))['annPoints']
        if hh < crop_size or ww < crop_size:
            imsave(os.path.join(image_save_path, image_file.split('.')[0] + "_%d.jpg" % i), img)
            scio.savemat(os.path.join(label_save_path, label_file.split('.')[0] + "_%d.mat" % i), {"annPoints": annPoints})
            continue
        h_pad = crop_size - hh % crop_size
        w_pad = crop_size - ww % crop_size
        h_slice = [crop_size * i for i in range(hh / crop_size + 1)]
        w_slice = [crop_size * i for i in range(ww / crop_size + 1)]
        h_border = [h_pad / (hh / crop_size)] * (hh / crop_size - 1) + [h_pad - h_pad / (hh / crop_size) * (hh / crop_size - 1)]
        w_border = [w_pad / (ww / crop_size)] * (ww / crop_size - 1) + [w_pad - w_pad / (ww / crop_size) * (ww / crop_size - 1)]
        h_border = [0] + np.cumsum(h_border).tolist()
        w_border = [0] + np.cumsum(w_border).tolist()
        # print h_pad, h_border, w_pad, w_border
        h_indexs = np.array(h_slice) - np.array(h_border)
        w_indexs = np.array(w_slice) - np.array(w_border)
        # print h_index, w_index
        sum_count = 0
        for i, (h_index, w_index) in enumerate(itertools.product(h_indexs, w_indexs)):
            patch = img[h_index:h_index + crop_size, w_index:w_index + crop_size,...]
            count = annPoints[((annPoints[:,0] >= w_index) & (annPoints[:,0] < w_index + crop_size) & \
                               (annPoints[:,1] >= h_index) & (annPoints[:,1] < h_index + crop_size) )]
            print w_index, w_index + crop_size, h_index, h_index + crop_size, count, '==' * 10 
            count[:, 0] = count[:, 0] - w_index
            count[:, 1] = count[:, 1] - h_index
            print patch.shape, np.size(count)
            sum_count += np.size(count)
            imsave(os.path.join(image_save_path, image_file.split('.')[0] + "_%d.jpg" % i), patch)
            scio.savemat(os.path.join(label_save_path, label_file.split('.')[0] + "_%d.mat" % i), {"annPoints": count})
        print img.shape, annPoints.shape, annPoints[:,0].max(), annPoints[:,1].min()
        print "*" * 30, sum_count