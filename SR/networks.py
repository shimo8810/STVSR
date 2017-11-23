'''
フレーム補間ネットワーク
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


class VGG16(chainer.Chain):
    '''
    特徴抽出としてのVGG16
    '''
    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

    def __call__(self, x):
        # Using max_pooling -> ave_pooling
        # 1 Layer
        h  = F.relu(self.conv1_1(x))
        h1 = F.relu(self.conv1_2(h))
        # 2 Layer
        # h  = F.max_pooling_2d(h1, ksize=2)
        h  = F.average_pooling_2d(h1, ksize=2)
        h  = F.relu(self.conv2_1(h))
        h2 = F.relu(self.conv2_2(h))
        # 3 Layer
        # h  = F.max_pooling_2d(h2, ksize=2)
        h = F.average_pooling_2d(h2, ksize=2)
        h  = F.relu(self.conv3_1(h))
        h  = F.relu(self.conv3_2(h))
        h3 = F.relu(self.conv3_3(h))
        # 4 Layer
        # h  = F.max_pooling_2d(h3, ksize=2)
        h = F.average_pooling_2d(h3, ksize=2)
        h  = F.relu(self.conv4_1(h))
        h  = F.relu(self.conv4_2(h))
        h4 = F.relu(self.conv4_3(h))
        # 5 Layer
        # h = F.max_pooling_2d(h4, ksize=2)
        h = F.average_pooling_2d(h4, ksize=2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h5 = F.relu(self.conv5_3(h))
        return h5
        # return h1, h2, h3, h4, h5


class VGG16Evaluator(chainer.Chain):
    def __init__(self, generator, vgg16):
        super(VGG16Evaluator,  self).__init__()
        self.y = None
        self.loss = None
        self.psnr = None
        self.vgg16 = vgg16
        with self.init_scope():
            self.generator = generator

    def __call__(self, x, t):
        self.y = None
        self.loss = None
        self.psnr = None
        self.y = self.generator(x)
        loss_mse = F.mean_squared_error(self.y, t)
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                y_cont = self.vgg16(self.y)
                t_cont = self.vgg16(t)
        loss_cont = F.mean_squared_error(y_cont, t_cont)
        self.loss = loss_cont + loss_mse
        self.psnr = 10 * F.log10(1.0 / loss_mse)
        reporter.report({'loss': self.loss, 'loss_cont':loss_cont, 'loss_mse':loss_mse, 'PSNR': self.psnr}, self)
        return self.loss

class SRCNN(chainer.Chain):
    def __init__(self, ch_scale=1, fil_sizes=(9,5,5)):
        init_w = chainer.initializers.HeNormal()
        super(SRCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch_scale * 32, ksize=fil_sizes[0], stride=1, pad=fil_sizes[0] // 2, initialW=init_w)
            self.conv2 = L.Convolution2D(None, ch_scale * 16, ksize=fil_sizes[1], stride=1, pad=fil_sizes[1] // 2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 1, ksize=fil_sizes[2], stride=1, pad=fil_sizes[2] // 2, initialW=init_w)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h

class VDSR(chainer.Chain):
    def __init__(self, depth=5):
        init_w = chainer.initializers.HeNormal()
        super(VDSR, self).__init__()
        with self.init_scope():
            self.conv_in = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
            self._forward_list = []
            for i in range(depth - 2):
                name = 'conv_{}'.format(i + 1)
                conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
                setattr(self, name, conv)
                self._forward_list.append(name)
            self.conv_out = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1, initialW=init_w)

    def __call__(self, x):
        h = F.relu(self.conv_in(x))
        for name in self._forward_list:
            l = getattr(self, name)
            h = F.relu(l(h))
        h = F.relu(self.conv_out(h))
        return h + x

class DRLSR(chainer.Chain):
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(DRLSR, self).__init__()
        with self.init_scope():
            self.conv1_3 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv1_5 = L.Convolution2D(None, 8, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv1_9 = L.Convolution2D(None, 8, ksize=9, stride=1, pad=4, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 16, ksize=1, stride=1, pad=0, initialW=init_w)
            self.conv22= L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv23= L.Convolution2D(None, 16, ksize=1, stride=1, pad=0, initialW=init_w)
            self.conv3_3 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv3_5 = L.Convolution2D(None, 8, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv3_9 = L.Convolution2D(None, 8, ksize=9, stride=1, pad=4, initialW=init_w)
            self.conv4 = L.Convolution2D(None, 1, ksize=1, stride=1, pad=0, initialW=init_w)

    def __call__(self, x):
        h = F.concat((F.relu(self.conv1_3(x)), \
                      F.relu(self.conv1_5(x)), \
                      F.relu(self.conv1_9(x))), axis=1)

        h = F.relu(self.conv2(h))
        h = F.relu(self.conv22(h))
        h = F.relu(self.conv23(h))

        h = F.concat((F.relu(self.conv3_3(h)),
                      F.relu(self.conv3_5(h)),
                      F.relu(self.conv3_9(h))), axis=1)

        h = F.relu(self.conv4(h))

        return h + x
