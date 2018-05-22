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
        return h1, h2, h3, h4, h5

class CAEFINet(chainer.Chain):
    def __init__(self, vgg_path='', f_size=5, n_ch=8, size=64):
        if size % 16 != 0:
            raise ValueError('size must be a multiple of 16.')
        init_w = chainer.initializers.HeNormal()
        super(CAEFINet, self).__init__()

        if vgg_path is not None:
            vgg16 = VGG16()
            chainer.serializers.load_npz(vgg_path, vgg16)
            vgg16.to_gpu()
            self.vgg16 = vgg16
        else:
            self.vgg16 = None

        with self.init_scope():
            # encoder
            self.enc_conv1 = L.Convolution2D(None, n_ch * 1, ksize=f_size,
                                stride=1, pad=f_size//2, initialW=init_w) # 64x64
            self.enc_conv2 = L.Convolution2D(None, n_ch * 2, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 32x32
            self.enc_conv3 = L.Convolution2D(None, n_ch * 4, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 16x16
            self.enc_conv4 = L.Convolution2D(None, n_ch * 8, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 8x8

            # decoder
            self.dec_conv1 = L.Deconvolution2D(None, n_ch * 4, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size//4, size//4)) # 16x16
            self.dec_conv2 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size//2, size//2)) # 32z32
            self.dec_conv3 = L.Deconvolution2D(None, n_ch * 1, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size, size)) # 64x64
            self.dec_conv4 = L.Convolution2D(None, 3, ksize=f_size,
                    stride=1, pad=f_size//2, initialW=init_w) # 64x64

    def encode(self, x):
        batch, f, c, w, h = x.shape
        h = F.reshape(x, (batch, f * c, w, h))
        h1 = F.relu(self.enc_conv1(h)) # 64x64
        h2 = F.relu(self.enc_conv2(h1)) # 32z32
        h3 = F.relu(self.enc_conv3(h2)) # 16x16
        h4 = F.relu(self.enc_conv4(h3)) # 8x8
        return h1, h2, h3, h4

    def decode(self, h1, h2, h3, h4):
        h = F.concat((F.relu(self.dec_conv1(h4)), h3), axis=1)
        h = F.concat((F.relu(self.dec_conv2(h)),  h2), axis=1)
        h = F.concat((F.relu(self.dec_conv3(h)),  h1), axis=1)
        return F.relu(self.dec_conv4(h))


    def __call__(self, x):
        h1, h2, h3, h4 = self.encode(x)
        return self.decode(h1, h2, h3, h4)

    def get_loss_func(self, weight=1.0, coef_decay='exp'):
        r = 0.1
        N = 5
        C = weight
        if coef_decay == 'exp':
            k = np.exp(np.log(r) / (N - 1) * np.arange(N))
            k = k / k.sum()
        elif coef_decay == 'lin':
            k = -1 * (1 - r) / (N - 1) * np.arange(N) + 1
            k = k / k.sum()
        else:
            raise ValueError("coef_decay parameter must be 'exp' or 'lin'")
        def lf(x, t):
            h1, h2, h3, h4 = self.encode(x)
            y = self.decode(h1, h2, h3, h4)
            mse_loss = F.mean_squared_error(y, t)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                y_content = self.vgg16(y)
                t_content = self.vgg16(t)
            cont_loss = 0.0
            k = [0.9, 0.7, 0.5, 0.3, 0.1]
            for i in range(len(y_content)):
                cont_loss += k[i] * F.mean_squared_error(y_content[i], t_content[i])
            self.mse_loss = mse_loss
            self.cont_loss = cont_loss
            self.loss = mse_loss + C * self.cont_loss
            self.psnr = 10 * F.log10(1.0 / mse_loss)
            chainer.report({'mse_loss': self.mse_loss, 'cont_loss':self.cont_loss,
                            'loss': self.loss, 'psnr':self.psnr}, observer=self)
            return self.loss
        return lf
