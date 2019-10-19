import os
import cv2
cv2.setNumThreads(0)
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import numpy as np
import random
import sys
import logging

from src.crowd_counting import CrowdCounter
from src import network

from src.timer import Timer
from src import utils
from src import density_gen

from src.datasets import datasets, CreateDataLoader

from src.train_options import TrainOptions

import torch.nn as nn

import src.ssim as ssim

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, opt):
    opt.logger.info(text)


logging.basicConfig(level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%d-%H:%M", 
            handlers=[
                logging.StreamHandler()
            ])


if __name__ == '__main__':
    
    rand_seed = 64678
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        random.seed(rand_seed)
    
    train_opt = TrainOptions()
    opt = train_opt.parse()
        
    vis_exp = train_opt.vis_exp
    data_loader_train = CreateDataLoader(opt, phase='train')

    loss_scale = opt.loss_scale

    momentum = 0.99
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    optimizer = lambda x: torch.optim.Adam(filter(lambda p: p.requires_grad, x.parameters()), lr=opt.lr)

    #load net and initialize it
    net = CrowdCounter(optimizer=optimizer, opt=opt)

    scheduler = None
    # scheduler = MultiStepLR(net.optimizer, milestones=range(opt.epochs)[::300][1:], gamma=0.1)
    # scheduler = MultiStepLR(net.optimizer, milestones=[1], gamma=0.1)


    net.train()

    #training configuration
    start_step = 0
    end_step = opt.epochs
    disp_interval = opt.disp_interval
    save_interval = opt.save_interval

    # training
    train_loss = 0
    step_cnt = 1
    re_cnt = False


    t = Timer()
    t.tic()

    print("Start training")
    for epoch in range(start_step, end_step+1):
        step = -1
        train_loss = 0
        outer_timer = Timer()
        outer_timer.tic()
        '''regenerate crop patches'''
        data_loader_train.shuffle_list()

        load_timer = Timer()
        load_time = 0.0
        iter_timer = Timer()
        iter_time = 0.0
        for i, datas in enumerate(\
                    DataLoader(data_loader_train, batch_size=opt.batch_size, \
                                                    shuffle=True, num_workers=4,drop_last=True)):
            step_cnt += 1
            if i != 0:
                load_time += load_timer.toc(average=False)
            iter_timer.tic()
            img_data = datas[0]
            gt_data = datas[1]
            raw_patch = datas[2]
            gt_count = datas[3]
            fnames = [data_loader_train.query_fname(i) for i in datas[4]]
            batch_size = len(fnames)

            step = step + 1

            net.train()
            density_map = net(img_data, gt_data)
            net.backward(loss_scale)

            loss_value = float(net.loss.item())
            train_loss += loss_value


            if step % disp_interval == 0 or \
                step_cnt % save_interval == 0:
                with torch.no_grad():
                    if step_cnt % save_interval == 0:
                        net.eval()
                        density_map_after = net(img_data)
                        density_map_after = density_map_after.detach().data.cpu().numpy()
                        net.train()
                    raw_patch = raw_patch.detach().data.cpu().numpy()
                    gt_data = gt_data.detach().data.cpu().numpy()
                    density_map = density_map.detach().data.cpu().numpy()

            ''' Display training loss and other train info'''
            if step % disp_interval == 0:            
                gt_count = np.sum(gt_data.reshape(batch_size, -1), axis=-1)
                et_count = np.sum(density_map.reshape(batch_size, -1), axis=-1)
                duration = t.toc(average=False)
                fps = disp_interval * batch_size / duration
                # utils.save_results(img_data,gt_data,density_map, opt.expr_dir, fname=blob['fname'], epoch=epoch)
                log_text = 'epoch: %04d,' % epoch + ' step %04d,' % step + ' Time: %.2fs,' % fps + \
                           ' gt_cnt: %s,' % "{}".format(["%.1f" % gt_count.max(), "%.1f" % gt_count.mean(), "%.1f" % gt_count.min()]) + \
                           ' et_cnt: %s,' % "{}".format(["%.1f" % et_count.max(), "%.1f" % et_count.mean(), "%.1f" % et_count.min()]) + \
                           ' loss: %e' % float(loss_value)

                log_print(log_text, opt)
                re_cnt = True
                if opt.use_tensorboard:
                    vis_exp.add_scalar_value('train_raw_loss', loss_value, step=step_cnt)


            ''' Save training image patch, and corresponding gt density map patch, 
                predicted density patch before and after loss backprop'''
            if step_cnt % save_interval == 0:
                for i in range(density_map.shape[0]):
                    density_gen.save_image(raw_patch[i], opt.expr_dir + './sup/', 'img_step%d_%d_0data.jpg' % (step_cnt, i))
                    density_gen.save_density_map(gt_data[i], opt.expr_dir + "./sup/", 'img_step%d_%d_1previous.jpg' % (step_cnt, i))
                    density_gen.save_density_map(density_map[i], opt.expr_dir + "./sup/", 'img_step%d_%d_2now.jpg' % (step_cnt, i))
                for i in range(density_map_after.shape[0]):
                    density_gen.save_density_map(density_map_after[i], opt.expr_dir + "./sup/", 'img_step%d_%d_3after.jpg' % (step_cnt, i))

            
            if re_cnt:
                t.tic()
                re_cnt = False
            iter_time += iter_timer.toc(average=False)
            load_timer.tic()

        duration = outer_timer.toc(average=False)
        logging.info("epoch {}: {} seconds; Path: {}".format(epoch, duration, opt.expr_dir))
        logging.info("load/iter/cuda: {} vs {} vs {} seconds; iter: {}".format(load_time, iter_time, net.cudaTimer.tot_time, net.cudaTimer.calls))
        net.cudaTimer.tot_time = 0


        save_name = os.path.join(opt.expr_dir, '%06d.h5' % epoch)
        network.save_net(save_name, net)

        if scheduler != None:
            scheduler.step()
            logging.info(scheduler.get_lr())

        logging.info("Train loss: {}".format(train_loss/data_loader_train.get_num_samples()))

        if opt.use_tensorboard:
            try:
                vis_exp.add_scalar_value('train_loss', train_loss/data_loader_train.get_num_samples(), step=epoch)
            except:
                pass

