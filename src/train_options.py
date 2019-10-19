import argparse
import os
import torch
from datetime import datetime
import shutil
import time
import random
import glob
import logging
import sys

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--gpus', type=str, help='gpu_id')
        parser.add_argument('--dataset', type=str, default='shanghaiA', help='dataset')


        parser.add_argument('--epochs', type=int, default=900)
        parser.add_argument('--lr', type=float, default=0.00001)

        parser.add_argument('--visual', dest='use_tensorboard', action='store_true')
        parser.add_argument('--no-visual', dest='use_tensorboard', action='store_false')
        parser.set_defaults(use_tensorboard=True)


        parser.add_argument('--save', dest='save_model_para', action='store_true')
        parser.add_argument('--no-save', dest='save_model_para', action='store_false')
        parser.set_defaults(save_model_para=True)

        parser.add_argument('--preload', dest='is_preload', action='store_true')
        parser.add_argument('--no-preload', dest='is_preload', action='store_false')
        parser.set_defaults(is_preload=True)

        parser.add_argument('--disp_interval', type=int, default=50)
        parser.add_argument('--save_interval', type=int, default=500)

        parser.add_argument('--batch_size', type=int, default=1)


        parser.add_argument('--pretrain', type=str)


        parser.add_argument('--crop_type', type=str, default="Fixed")
        parser.add_argument('--crop_scale', type=int, default=4)
        parser.add_argument('--crop_size', type=str, default='224x224')
        parser.add_argument('--patches_per_sample', type=int, default=5)


        parser.add_argument('--loss', type=str, default="MSE")
        parser.add_argument('--loss_scale', type=float, default=1.0)
        parser.add_argument('--model_name', type=str)

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        opt.logger.info(message)

        file_name = os.path.join(opt.expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        model = opt.model_name
        dataset_name = opt.dataset #dataset name - used for saving model file
        exp = 'v7-{}-{}-{}/'.format(dataset_name, model, datetime.now().strftime('exp-%m-%d_%H-%M'))
        expr_dir = './saved_models/{}/'.format(exp) #model files are saved here

        opt.crop_size = map(int, opt.crop_size.split('x'))

        if opt.save_model_para and not os.path.exists(expr_dir):
            os.mkdir(expr_dir)
            os.mkdir(expr_dir+'./sup/')

        else:
            expr_dir = '../../temp1/'
            if not os.path.exists(expr_dir+'./sup/'):
                os.mkdir(expr_dir+'./sup/')

        opt.expr_dir = expr_dir

        logger = logging.getLogger()
        fh = logging.FileHandler("{0}/{1}.log".format(expr_dir, 'log'), mode='w')
        fh.setFormatter(logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="%d-%H:%M"))
        logger.addHandler(fh)
        opt.logger =  logger

        self.opt = opt

        #Tensorboard  config
        use_tensorboard = opt.use_tensorboard
        remove_all_log = False   # remove all historical experiments in TensorBoardO
        use_tensorboard = use_tensorboard and CrayonClient is not None
        self.vis_exp = None

        if use_tensorboard:
            cc = CrayonClient(hostname='8.8.8.8', port=7879)
            if remove_all_log:
                cc.remove_all_experiments()
            random.seed(time.time())
            vis_exp_name = exp + str(random.random())
            opt.vis_exp_name = vis_exp_name
            self.vis_exp = cc.create_experiment(vis_exp_name)

        import socket
        hostname = socket.gethostname()

        # set gpu ids
        str_ids = opt.gpus.split(',')
        opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpus.append(id)
        if len(opt.gpus) > 0:
            torch.cuda.set_device(opt.gpus[0])

        self.opt = opt
        self.print_options(opt)

        return self.opt
