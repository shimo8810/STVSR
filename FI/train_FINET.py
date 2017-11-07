'''
train SRCNN Network
simple network
'''
import os
from os import path
import argparse
import random
import csv
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

#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

DATA_PATH = '/media/shimo/HDD_storage/DataSet'
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


class GenEvaluator(chainer.Chain):
    def __init__(self, generator):
        super(GenEvaluator, self).__init__()
        self.y = None
        self.loss = None
        self.psnr = None

        with self.init_scope():
            self.generator = generator

    def __call__(self, x, t):
        self.y = None
        self.loss = None
        self.psnr = None
        self.y = self.generator(x)
        self.loss = F.mean_squared_error(self.y, t)
        self.psnr = 10 * F.log10(1.0 / self.loss)
        reporter.report({'loss': self.loss, 'PSNR': self.psnr}, self)
        return self.loss


class FINET(chainer.Chain):
    def __init__(self, ch_scale=1, fil_sizes=(9, 5, 5)):
        init_w = chainer.initializers.HeNormal()
        super(FINET, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 16, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h


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
    parser.add_argument('--ch_scale', '-c', type=int, default=1,
                        help='ch scale')
    parser.add_argument('--fil_sizes', '-f', type=int, nargs='+', default=[9, 5, 5],
                        help='filter(kernel) sizes')
    parser.add_argument('--iter_parallel', action='store_true', default=False,
                        help='filter(kernel) sizes')
    args = parser.parse_args()

    # parameter出力
    print("-=Learning Parameter=-")
    print("# Max Epochs: {}".format(args.epoch))
    print("# Batch Size: {}".format(args.batchsize))
    print("# Learning Rate: {}".format(args.learnrate))
    print('# Train Dataet: General 100')
    if args.iter_parallel:
        print("# Data Iters that loads in Parallel")
    print("\n")

    # 保存ディレクトリ
    # save didrectory
    outdir = path.join(
        ROOT_PATH, 'results/FINET'.format(args.ch_scale, *args.fil_sizes))
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    print('# loading dataet(General100, Set14) ...')
    train = SequenceDataset(dataset='train')
    test = SequenceDataset(dataset='test')
   # prepare model
    model = GenEvaluator(FINET())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # setup iter
    if args.iter_parallel:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=8)
        test_iter = chainer.iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False, n_processes=8)
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
    trainer.extend(extensions.ExponentialShift(
        "lr", 0.1), trigger=(100, 'epoch'))
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
        ['epoch', 'main/loss', 'validation/main/loss', 'main/PSNR', 'validation/main/PSNR', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
