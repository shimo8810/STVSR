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
        self.loss = 10 * loss_cont + loss_mse
        self.psnr = 10 * F.log10(1.0 / loss_mse)
        reporter.report({'loss': self.loss, 'loss_cont':loss_cont, 'loss_mse':loss_mse, 'PSNR': self.psnr}, self)
        return self.loss




class FINET(chainer.Chain):
    '''
    単純な3層構造
    PSNR:25~6程度
    モーションブラー(もとい入力のズレ)が発生する
    '''
    def __init__(self):
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

class FINet(chainer.Chain):
    '''
    単純な3層構造
    PSNR:25~6程度
    モーションブラー(もとい入力のズレ)が発生する
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(FINet, self).__init__()
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

class FINET2(chainer.Chain):
    '''
    凹ませてみた,res層なし
    PSNR:22~23
    カス
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(FINET2, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv4 = L.Deconvolution2D(None, 64, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv5 = L.Deconvolution2D(None, 32, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv6 = L.Deconvolution2D(None, 16, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv7 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.dconv4(h))
        h = F.relu(self.dconv5(h))
        h = F.relu(self.dconv6(h))
        h = F.relu(self.conv7(h))
        return h

class FINet3(chainer.Chain):
    '''
    単純な3層構造
    PSNR:25~6程度
    モーションブラー(もとい入力のズレ)が発生する
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(FINet3, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv1_2 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1, initialW=init_w)

    def __call__(self, x):
        h1 = F.relu(self.conv1_1(x[:, 0, :, :, :]))
        h2 = F.relu(self.conv1_2(x[:, 1, :, :, :]))
        h = F.concat((h1, h2), axis=1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h

class ResFINet(chainer.Chain):
    '''
    FINet2にresを加えてみたヤツ
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(ResFINet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv4 = L.Deconvolution2D(None, 32, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv5 = L.Deconvolution2D(None, 16, ksize=5, stride=1, pad=0, initialW=init_w)
            self.dconv6 = L.Deconvolution2D(None, 8, ksize=5, stride=1, pad=0, initialW=init_w)
            self.conv7 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h0 = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h0))
        h2 = F.relu(self.conv2(h1))
        h = F.relu(self.conv3(h2))
        h = F.relu(self.dconv4(h)) + h2
        h = F.relu(self.dconv5(h)) + h1
        h = F.relu(self.dconv6(h))
        h = F.relu(self.conv7(h))
        return h

class ResFINet2(chainer.Chain):
    '''
    FINet2にresを加えてみたヤツ
    ストライドを増やして構造を圧縮しているよ
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(ResFINet2, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv4 = L.Convolution2D(None, 64, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv5 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
            self.dconv6 = L.Deconvolution2D(None, 32, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv7 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.dconv8 = L.Deconvolution2D(None, 16, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h))
        h4 = F.relu(self.conv4(h3))
        h = F.relu(self.conv5(h4)) + h4
        h = F.relu(self.dconv6(h)) + h3
        h = F.relu(self.conv7(h))
        h = F.relu(self.dconv8(h)) + h1
        h = F.relu(self.conv9(h))
        return h

class ResFINet3(chainer.Chain):
    '''
    ResFINet2圧縮させないネットワークを加える
    ストライドを増やして構造を圧縮しているよ
    '''
    def __init__(self):
        init_w = chainer.initializers.HeNormal()
        super(ResFINet3, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv3 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv4 = L.Convolution2D(None, 64, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv5 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
            self.dconv6 = L.Deconvolution2D(None, 32, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv7 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.dconv8 = L.Deconvolution2D(None, 16, ksize=3, stride=2, pad=1, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)

            self.conv = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1, initialW=init_w)

            self.conv10 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv11 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv12 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        hh = F.relu(self.conv10(h))
        hh = F.relu(self.conv11(hh))
        hh = F.relu(self.conv12(hh))

        h1 = F.relu(self.conv1(h))

        h = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h))
        h4 = F.relu(self.conv4(h3))
        h = F.relu(self.conv5(h4)) + h4
        h = F.relu(self.dconv6(h)) + h3
        h = F.relu(self.conv7(h))
        h = F.relu(self.dconv8(h)) + h1
        h = F.relu(self.conv9(h))

        h = F.concat((hh, h), axis=1)

        h = F.relu(self.conv(h))
        return h

class DeepFINet(chainer.Chain):
    '''
    すごく深い
    args:
        depth: 層の数
    '''
    def __init__(self, depth=3):
        init_w = chainer.initializers.HeNormal()
        super(DeepFINet, self).__init__()
        with self.init_scope():
            self.conv_in = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self._forward_list = []
            for i in range(depth - 2):
                name = 'conv_{}'.format(i + 1)
                conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
                setattr(self, name, conv)
                self._forward_list.append(name)
            self.conv_out = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h = F.relu(self.conv_in(h))
        for name in self._forward_list:
            l = getattr(self, name)
            h = F.relu(l(h))
        h = F.relu(self.conv_out(h))
        return h

class DeepResFINet(chainer.Chain):
    '''
    すごく深い+Resにしてある
    args:
        depth: 層の数
    '''
    def __init__(self, depth=3):
        self.depth = depth
        init_w = chainer.initializers.HeNormal()
        super(DeepResFINet, self).__init__()
        with self.init_scope():
            self.conv_in = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1, initialW=init_w)
            self._forward_list = []
            for i in range(depth - 2):
                name = 'conv_{}'.format(i + 1)
                conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=init_w)
                setattr(self, name, conv)
                self._forward_list.append(name)
            self.conv_out = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h = F.relu(self.conv_in(h))
        for name in self._forward_list:
            l = getattr(self, name)
            h = F.relu(l(h)) + h
        h = F.relu(self.conv_out(h))
        return h

class AEFINet(chainer.Chain):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h)) # 8 , H, W
        h  = F.relu(self.conv_down2(h1)) # 16, H/2, W/2
        h2 = F.relu(self.conv3(h)) # 16, H/2, W/2
        h  = F.relu(self.conv_down4(h2)) # 32, H/4, W/4
        h = F.relu(self.conv5(h)) + h # 32, H/4, W/4
        h = F.relu(self.conv_up6(h)) + h2 # 16, H/2, W/2
        h = F.relu(self.conv7(h)) # 16, H/2, W/2
        h = F.relu(self.conv_up8(h)) + h1 # 8 , H, W
        return F.relu(self.conv9(h))

class AEFINetBN(chainer.Chain):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    BNありバージョン
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINetBN, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)
            self.norm1 = L.BatchNormalization(n_ch)
            self.norm2 = L.BatchNormalization(n_ch * 2)
            self.norm3 = L.BatchNormalization(n_ch * 2)
            self.norm4 = L.BatchNormalization(n_ch * 4)
            self.norm5 = L.BatchNormalization(n_ch * 4)
            self.norm6 = L.BatchNormalization(n_ch * 2)
            self.norm7 = L.BatchNormalization(n_ch * 2)
            self.norm8 = L.BatchNormalization(n_ch)
            self.norm9 = L.BatchNormalization(3)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.norm1(self.conv1(h))) # 8 , H, W
        h  = F.relu(self.norm2(self.conv_down2(h1))) # 16, H/2, W/2
        h2 = F.relu(self.norm3(self.conv3(h))) # 16, H/2, W/2
        h  = F.relu(self.norm4(self.conv_down4(h2))) # 32, H/4, W/4
        h = F.relu(self.norm5(self.conv5(h))) + h # 32, H/4, W/4
        h = F.relu(self.norm6(self.conv_up6(h))) + h2 # 16, H/2, W/2
        h = F.relu(self.norm7(self.conv7(h))) # 16, H/2, W/2
        h = F.relu(self.norm8(self.conv_up8(h))) + h1 # 8 , H, W
        return F.relu(self.norm9(self.conv9(h)))

class VAEFINet(chainer.Chain):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2, latent_size=1000):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINet, self).__init__()

        with self.init_scope():
            # Encode
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)

            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            # Decode
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

            self.mu = L.Linear(None, latent_size)
            self.ln_var = L.Linear(None, latent_size)


    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        # encoding
        h1 = F.relu(self.conv1(h)) #output 8 , H, W
        h  = F.relu(self.conv_down2(h1)) #output 16, H/2, W/2
        h2 = F.relu(self.conv3(h)) #output 16, H/2, W/2
        h  = F.relu(self.conv_down4(h2)) #output 32, H/4, W/4

        h = F.relu(self.conv5(h)) #output 32, H/4, W/4
        #decoding
        h = F.relu(self.conv_up6(h)) + h2 #output 16, H/2, W/2
        h = F.relu(self.conv7(h)) #output 16, H/2, W/2
        h = F.relu(self.conv_up8(h)) + h1 #output 8 , H, W
        return F.relu(self.conv9(h))

    def encode(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h)) #output 8 , H, W
        h  = F.relu(self.conv_down2(h1)) #output 16, H/2, W/2
        h2 = F.relu(self.conv3(h)) #output 16, H/2, W/2
        h  = F.relu(self.conv_down4(h2)) #output 32, H/4, W/4
        mu = self.mu(h)
        var = self.ln_var(h)
        return mu, var

    def get_loss_func(self, k=1):
        '''
        return loss func of vae
        '''
        def loss(x, y):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            #
            # reconstruction loss
            rec_loss = F.mean_squared_error()
            return self.loss
