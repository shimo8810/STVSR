'''
training FI Networks
most simple network
'''
import os
from os import path
import argparse
import random
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import h5py
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.datasets import (TupleDataset, TransformDataset)
from chainer.links.model.vision import resnet
from chainercv import transforms
# self file
import networks
import my_datasets

# Path config
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

def load_dataset():
    '''
    load dataset
    '''
    train = h5py.File(path.join(ROOT_PATH, 'dataset/SCENE10_2K_60fps.hdf5'))
    test = h5py.File(path.join(ROOT_PATH, 'dataset/SCENE1_2K_mini.hdf5'))
    train_x, train_y = np.array(train['x_data']), np.array(train['y_data'])
    test_x, test_y = np.array(test['x_data']), np.array(test['y_data'])

    train = TupleDataset(train_x, train_y)
    test = TupleDataset(test_x, test_y)

    return train, test
def main():
    '''
    main function
    start point
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='SCENE1_mini',
                        help='')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    outdir = path.join(ROOT_PATH, 'results/Only_FI')
    if not path.exists(outdir):
        os.makedirs(outdir)

    #loading dataset
    print('# loading {}'.format(args.dataset))
    train, test = load_dataset()

    # prepare model
    model = networks.ImageEvaluator(networks.SimpleFI())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate)
    optimizer.setup(model)

    # setup iter
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # eval test data
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

    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
