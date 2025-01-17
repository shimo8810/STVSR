'''
train SRCNN Network
simple network
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


def transform(data):
    x_img, y_img = data

    x_flip, y_flip, rot = np.random.choice([True, False], 3)
    # random flip
    x_img = transforms.flip(x_img, y_flip=y_flip, x_flip=x_flip)
    y_img = transforms.flip(y_img, y_flip=y_flip, x_flip=x_flip)

    # # random rot
    if rot:
        x_img = np.rot90(x_img, axes=(-2, -1))
        y_img = np.rot90(y_img, axes=(-2, -1))

    return x_img, y_img


def load_dataset():
    train = h5py.File(path.join(ROOT_PATH, 'dataset/General100_train.hdf5'))
    test = h5py.File(path.join(ROOT_PATH, 'dataset/Set14_test.hdf5'))

    train_x, train_y = np.array(train['x_data']) / 255, np.array(train['y_data']) / 255
    test_x, test_y = np.array(test['x_data']) / 255, np.array(test['y_data']) / 255

    train = TupleDataset(train_x, train_y)
    test = TupleDataset(test_x, test_y)

    train = TransformDataset(train, transform)

    return train, test


def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--depth', '-d', type=int, default=5,
                        help='depth para')
    parser.add_argument('--iter_parallel', action='store_true', default=False,
                        help='filter(kernel) sizes')
    parser.add_argument('--opt', '-o', default='sgd',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    # parameter出力
    print("-=Learning Parameter=-")
    print("# Max Epochs: {}".format(args.epoch))
    print("# Batch Size: {}".format(args.batchsize))
    print("# Learning Rate: {}".format(args.learnrate))
    print("# Number of Depth: {}".format(args.depth))
    print('# Train Dataet: General 100')
    print('# Test Dataet: Set 14')
    if args.iter_parallel:
        print("# Data Iters that loads in Parallel")
    print("\n")

    # 保存ディレクトリ
    # save didrectory
    outdir = path.join(
        ROOT_PATH, 'results/VDSR_Depth_{}_opt_{}_lr_{}'.format(args.depth, args.opt, args.learnrate))
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    print('# loading dataet(General100, Set14) ...')
    train, test = load_dataset()

   # prepare model
    model = N.GenEvaluator(N.VDSR(depth=args.depth))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    if args.opt == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=args.learnrate)
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

    # eval test data
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    # dump loss graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # lr shift
    if args.opt == 'sgd':
        trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=(100, 'epoch'))
    elif args.opt == 'adam':
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
        ['epoch', 'main/loss', 'validation/main/loss', 'main/PSNR', 'validation/main/PSNR', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
