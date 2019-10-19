import os
import cv2
import debug
import torch
import numpy as np
from src.crowd_counting import CrowdCounter
from src import network
from src.RawLoader import ImageDataLoader, basic_config
from src import utils
import argparse
from src.sampler import basic_config as sampler_config
from src.sampler import mode_func as sampler_func
import torchvision.transforms as transforms
from src.datasets import datasets, CreateDataLoader
import src.density_gen as dgen
from src.timer import Timer
import itertools
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

#test data and model file path
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--gpus', type=str, help='gpu_id')
parser.add_argument('--dataset', type=str)
parser.add_argument('--prefix', type=str)

parser.add_argument('--preload', dest='is_preload', action='store_true')
parser.add_argument('--no-preload', dest='is_preload', action='store_false')
parser.set_defaults(is_preload=True)

parser.add_argument('--wait', dest='is_wait', action='store_true')
parser.add_argument('--no-wait', dest='is_wait', action='store_false')
parser.set_defaults(is_wait=True)

parser.add_argument('--save', dest='save_output', action='store_true')
parser.add_argument('--no-save', dest='save_output', action='store_false')
parser.set_defaults(save_output=False)

# crop adap
parser.add_argument('--test_fixed_size', type=int, default=-1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--epoch', type=int)

parser.add_argument('--name', type=str)
parser.add_argument('--split', type=str)


def test_model_origin(net, data_loader, save_output=False, save_path=None, test_fixed_size=-1, test_batch_size=1, gpus=None):
    timer = Timer()
    timer.tic()
    net.eval()
    mae = 0.0
    mse = 0.0
    detail = ''
    if save_output:
        print save_path
    for i, blob in enumerate(data_loader.get_loader(test_batch_size)):
        if (i * len(gpus) + 1) % 100 == 0:
            print "testing %d" % (i + 1)
        if save_output:
            index, fname, data, mask, gt_dens, gt_count = blob
        else:
            index, fname, data, mask, gt_count = blob

        with torch.no_grad():
            dens = net(data)
            if save_output:
                image = data.squeeze_().mul_(torch.Tensor([0.229,0.224,0.225]).view(3,1,1))\
                                        .add_(torch.Tensor([0.485,0.456,0.406]).view(3,1,1)).data.cpu().numpy()
                                        
                dgen.save_image(image.transpose((1,2,0))*255.0, save_path, fname[0].split('.')[0] + "_0_img.png")
                gt_dens = gt_dens.data.cpu().numpy()
                density_map = dens.data.cpu().numpy()
                dgen.save_density_map(gt_dens.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt.png")
                dgen.save_density_map(density_map.squeeze(), save_path, fname[0].split('.')[0] + "_2_et.png")
                _gt_count = gt_dens.sum().item()
                del gt_dens
        gt_count = gt_count.item()
        et_count = dens.sum().item()

        del data, dens
        detail += "index: {}; fname: {}; gt: {}; et: {};\n".format(i, fname[0].split('.')[0], gt_count, et_count)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
    mae = mae/len(data_loader)
    mse = np.sqrt(mse/len(data_loader))
    duration = timer.toc(average=False)
    print "testing time: %d" % duration
    return mae,mse,detail



def test_model_patches(net, data_loader, save_output=False, save_path=None, test_fixed_size=-1, test_batch_size=1, gpus=None):
    timer = Timer()
    timer.tic()
    net.eval()
    mae = 0.0
    mse = 0.0
    detail = ''
    if save_output:
        print save_path
    for i, blob in enumerate(data_loader.get_loader(1)):

        if (i + 1) % 10 == 0:
            print "testing %d" % (i + 1)
        if save_output:
            index, fname, data, mask, gt_dens, gt_count = blob
        else:
            index, fname, data, mask, gt_count = blob

        data = data.squeeze_()
        if len(data.shape) == 3:
            'image small than crop size'
            data = data.unsqueeze_(dim=0)
        mask = mask.squeeze_()
        num_patch = len(data)
        batches = zip([i * test_batch_size for i in range(num_patch // test_batch_size + int(num_patch % test_batch_size != 0))], 
                            [(i + 1) * test_batch_size for i in range(num_patch // test_batch_size)] + [num_patch])
        with torch.no_grad():
            dens_patch = []
            for batch in batches:
                bat = data[slice(*batch)]
                dens = net(bat).cpu()
                dens_patch += [dens]

            if args.test_fixed_size != -1:
                H, W = mask.shape
                _, _, fixed_size = data[0].shape
                assert args.test_fixed_size == fixed_size
                density_map = torch.zeros((H, W))
                for dens_slice, (x, y) in zip(itertools.chain(*dens_patch), itertools.product(range(W / fixed_size), range(H / fixed_size))):
                    density_map[y * fixed_size:(y + 1) * fixed_size, x * fixed_size:(x + 1) * fixed_size] = dens_slice
                H = mask.sum(dim=0).max().item()
                W = mask.sum(dim=1).max().item()
                density_map = density_map.masked_select(mask).view(H, W)
            else:
                density_map = dens_patch[0]

            gt_count = gt_count.item()
            et_count = density_map.sum().item()

            if save_output:
                image = data.mul_(torch.Tensor([0.229,0.224,0.225]).view(3,1,1))\
                                        .add_(torch.Tensor([0.485,0.456,0.406]).view(3,1,1))


                if args.test_fixed_size != -1:
                    H, W = mask.shape
                    _, _, fixed_size = data[0].shape
                    assert args.test_fixed_size == fixed_size
                    inital_img = torch.zeros((3, H, W))
                    for img_slice, (x, y) in zip(image, itertools.product(range(W / fixed_size), range(H / fixed_size))):
                        inital_img[:, y * fixed_size:(y + 1) * fixed_size, x * fixed_size:(x + 1) * fixed_size] = img_slice
                    H = mask.sum(dim=0).max().item()
                    W = mask.sum(dim=1).max().item()
                    inital_img = inital_img.masked_select(mask).view(3, H, W)
                    image = inital_img


                image = image.data.cpu().numpy()
                dgen.save_image(image.transpose((1,2,0))*255.0, save_path, fname[0].split('.')[0] + "_0_img.png")
                gt_dens = gt_dens.data.cpu().numpy()
                density_map = density_map.data.cpu().numpy()
                dgen.save_density_map(gt_dens.squeeze(), save_path, fname[0].split('.')[0] + "_1_gt.png")
                dgen.save_density_map(density_map.squeeze(), save_path, fname[0].split('.')[0] + "_2_et.png")
                del gt_dens
            del data, dens

        detail += "index: {}; fname: {}; gt: {}; et: {};\n".format(i, fname[0].split('.')[0], gt_count, et_count)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
    mae = mae/len(data_loader)
    mse = np.sqrt(mse/len(data_loader))
    duration = timer.toc(average=False)
    print "testing time: %d" % duration
    return mae,mse,detail




if __name__ == '__main__':
    args = parser.parse_args()
    # set gpu ids
    str_ids = args.gpus.split(',')

    args.gpus = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpus.append(id)
    if len(args.gpus) > 0:
        torch.cuda.set_device(args.gpus[0])
    args.loss = None
    args.test_crop_type = 'Adap'
    args.pretrain = None

    data_loader_test = CreateDataLoader(args, phase='test')


    optimizer = lambda x: torch.optim.Adam(filter(lambda p: p.requires_grad, x.parameters()))
    net = CrowdCounter(optimizer=optimizer, opt=args)

    if args.model_path.endswith('.h5'):
        output_path = args.model_path[:-3] + '/output/'
        if not os.path.exists(args.model_path[:-3]):
            os.mkdir(args.model_path[:-3])

        test_once = True
    else:
        output_path = args.model_path + '/output/'
        test_once = False

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if test_once:
        model_files = [args.model_path]
    elif args.epoch is not None:
        model_files = ['%06d.h5' % args.epoch]
        assert args.save_output
    elif not args.is_wait:
        def list_dir(watch_path):
            return itertools.chain(*[[filename] if (os.path.isfile(os.path.join(watch_path,filename)) and '.h5' in filename)\
                               else []\
                               for filename in os.listdir(watch_path)])

        model_files = list(list_dir(args.model_path))
        model_files.sort()
        model_files = model_files[::-1]
        assert not args.save_output

    else:
        model_files = ['%06d.h5' % epoch for epoch in range(0, 301)]
        assert not args.save_output

    if args.split is not None:
        model_files = ['%06d.h5' % epoch for epoch in map(int, args.split[:-1].split(','))]
        
    print model_files
    for model_file in model_files:

        epoch = model_file.split('.')[0] if not test_once else '0'

        output_dir = os.path.join(output_path, epoch)
        file_results = os.path.join(output_dir,'results.txt')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_dir = os.path.join(output_dir, 'density_maps')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        trained_model = os.path.join(args.model_path, epoch + '.h5') if not test_once else args.model_path

        while(not os.path.isfile(trained_model)):
            time.sleep(3)

        network.load_net(trained_model, net)

        if args.test_batch_size != 1 or args.test_fixed_size != -1:
            test_mae, test_mse, detail = test_model_patches(net, data_loader_test, args.save_output, \
                    output_dir, test_fixed_size=args.test_fixed_size, test_batch_size=args.test_batch_size, \
                    gpus=args.gpus)
        else:
            test_mae, test_mse, detail = test_model_origin(net, data_loader_test, args.save_output, \
                    output_dir, test_fixed_size=args.test_fixed_size, test_batch_size=args.test_batch_size, \
                    gpus=args.gpus)

        
        log_text = 'TEST EPOCH: %s, MAE: %.2f, MSE: %0.2f' % (epoch, test_mae, test_mse)

        print log_text

        with open(file_results, 'w') as f: 
            f.write(detail + 'MAE: %0.2f, MSE: %0.2f' % (test_mae, test_mse))
