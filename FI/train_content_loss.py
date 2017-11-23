'''
train SRCNN Network
simple network
'''
import os
from os import path
import argparse
import random
import csv
from tqdm import tqdm
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import h5py
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import (reporter, training)
from chainer.training import extensions
from chainer.datasets import (TupleDataset, TransformDataset)
from chainer.links.model.vision import resnet
from chainercv import transforms

import networks as N

#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

# DATA_PATH = '/media/shimo/HDD_storage/DataSet'
DATA_PATH = path.join(ROOT_PATH, 'dataset')


class SequenceDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='train'):
        self.image_paths = []
        csv_path = None
        if dataset == 'train':
            csv_path = 'Train_Mini_UCF101/train_data_loc.csv'
        elif  dataset == 'test':
            csv_path = 'Test_Mini_UCF101/train_data_loc.csv'

        with open(path.join(DATA_PATH, csv_path)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.image_paths.append(path.join(DATA_PATH, row[0]))

    def __len__(self):
        return len(self.image_paths)

    def get_example(self, i):
        data = np.load(self.image_paths[i])
        x_data = data['x_data']
        y_data = data['y_data']
        return x_data, y_data

class SequenceDatasetOnMem(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='train'):
        self.image_paths = []
        csv_path = None
        if dataset == 'train':
            csv_path = 'Train_Mini_UCF101/train_data_loc.csv'
        elif  dataset == 'test':
            csv_path = 'Test_Mini_UCF101/train_data_loc.csv'

        with open(path.join(DATA_PATH, csv_path)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.image_paths.append(path.join(DATA_PATH, row[0]))

        data = np.load(self.image_paths[0])
        nf, ch, h, w = data['x_data'].shape
        self.x_data = np.zeros((len(self.image_paths), nf, ch, h, w), dtype=np.float32)
        ch, h, w = data['y_data'].shape
        self.y_data = np.zeros((len(self.image_paths), ch, h, w), dtype=np.float32)

        print("loading datasaet {} ...".format(dataset))
        for i, p in tqdm(enumerate(self.image_paths)):
            data = np.load(p)
            self.x_data[i] = data['x_data']
            self.y_data[i] = data['y_data']

    def __len__(self):
        return len(self.image_paths)

    def get_example(self, i):
        return self.x_data[i], self.y_data[i]

def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--iter_parallel', '-p', action='store_true', default=False,
                        help='filter(kernel) sizes')
    parser.add_argument('--opt' , '-o', type=str, choices=('adam', 'sgd') ,default='adam')
    parser.add_argument('--depth', '-d', type=int, default=3,
                        help='DeepFINet Layer Depth')
    args = parser.parse_args()

    # parameter出力
    print("-=Learning Parameter=-")
    print("# Max Epochs: {}".format(args.epoch))
    print("# Batch Size: {}".format(args.batchsize))
    print("# Learning Rate: {}".format(args.learnrate))
    print("# Optimizer Method: {}".format(args.opt))
    print('# Train Dataet: General 100')
    if args.iter_parallel:
        print("# Data Iters that loads in Parallel")
    print("\n")

    # 保存ディレクトリ
    # save didrectory
    outdir = path.join(
        ROOT_PATH, 'results/FINet_VGG_content_loss_opt_{}'.format(args.opt))
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    print('# loading dataet(General100_train, General100_test) ...')
    if args.iter_parallel:
        train = SequenceDataset(dataset='train')
        test = SequenceDataset(dataset='test')
    else:
        train = SequenceDatasetOnMem(dataset='train')
        test = SequenceDatasetOnMem(dataset='test')

   # prepare model
    vgg16 = N.VGG16()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        vgg16.to_gpu()
    chainer.serializers.load_npz(path.join(ROOT_PATH, 'models/VGG16.npz'), vgg16)
    model = N.VGG16Evaluator(N.DeepFINet(depth=args.depth), vgg16)
    if args.gpu >= 0:
        model.to_gpu()

    # setup optimizer
    if args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.opt == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # setup iter
    if args.iter_parallel:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=6)
        test_iter = chainer.iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False, n_processes=6)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test, args.batchsize, repeat=False, shuffle=False)

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # # eval test data
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    # dump loss graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # lr shift
    if args.opt == 'sgd':
        trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=(100, 'epoch'))
    if args.opt == 'adam':
        trainer.extend(extensions.ExponentialShift("alpha", 0.1), trigger=(100, 'epoch'))
    # save snapshot
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_snapshot_{.updater.epoch}'), trigger=(10, 'epoch'))
    # log report
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    #  plot loss graph
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    # plot acc graph
    trainer.extend(
        extensions.PlotReport(
            ['main/PSNR', 'validation/main/PSNR'],
            'epoch', file_name='PSNR.png'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss','main/loss_mse', 'main/loss_cont', 'main/PSNR', 'validation/main/PSNR', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
