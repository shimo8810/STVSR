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

#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

class GenEvaluator(chainer.Chain):
    def __init__(self, generator):
        super(GenEvaluator, self).__init__()
        self.y = None
        self.loss = None
        self.acc = None

        with self.init_scope():
            self.generator = generator

    def __call__(self, x, t):
        self.y = None
        self.loss = None
        self.acc = None
        self.y = self.generator(x)
        self.loss = F.mean_squared_error(self.y, t)
        reporter.report({'loss': self.loss, 'accuracy': self.acc}, self)
        return self.loss


class SRCNN(chainer.Chain):
    def __init__(self):
        super(SRCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h

def transform(data):
    x_img, y_img = data

    x_flip, y_flip, rot, noise = np.random.choice([True, False], 4)
    # random flip
    x_img = transforms.flip(x_img, y_flip=y_flip, x_flip=x_flip)
    y_img = transforms.flip(y_img, y_flip=y_flip, x_flip=x_flip)

    # # random rot
    # if rot:
    #     x_img = np.rot90(x_img)
    #     y_img = np.rot90(y_img)

    # # random noise
    # if noise:
    #     transforms.pca_lighting(x_img, 0.1)
    return x_img, y_img

def load_dataset():
    train = h5py.File(path.join(ROOT_PATH, 'dataset/General100_train.hdf5'))
    test = h5py.File(path.join(ROOT_PATH, 'dataset/Set14_test.hdf5'))

    train_x, train_y = np.array(train['x_data']) / 255, np.array(train['y_data']) /255
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
    parser.add_argument('--learnrate', '-l', type=float, default=0.001,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    args = parser.parse_args()

    # 保存ディレクトリ
    # save didrectory
    outdir = path.join(ROOT_PATH, 'results/SRCNN')
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    print('# loading dataet( General100, Set14) ...')
    train, test = load_dataset()

   # prepare model
    model = GenEvaluator(SRCNN())
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
    # trainer.extend(
    #     extensions.PlotReport(
    #         ['main/accuracy', 'validation/main/accuracy'],
    #         'epoch', file_name='accuracy.png'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
