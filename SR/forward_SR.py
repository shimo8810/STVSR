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


class SRCNN(chainer.Chain):
    def __init__(self, ch_scale=1, fil_sizes=(9,5,5)):
        super(SRCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch_scale * 32, ksize=fil_sizes[0], stride=1, pad=fil_sizes[0] // 2)
            self.conv2 = L.Convolution2D(None, ch_scale * 16, ksize=fil_sizes[1], stride=1, pad=fil_sizes[1] // 2)
            self.conv3 = L.Convolution2D(None, 1, ksize=fil_sizes[2], stride=1, pad=fil_sizes[2] // 2)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
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

    args = parser.parse_args()

    # parameter出力

    # 保存ディレクトリ
    # save didrectory

   # prepare model
    model = GenEvaluator(SRCNN(ch_scale=args.ch_scale, fil_sizes=args.fil_sizes))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()



if __name__ == '__main__':
    main()
