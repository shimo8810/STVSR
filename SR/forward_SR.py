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

from scipy.misc import imresize
from PIL import Image
import numpy as np
import h5py
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import (reporter, training, serializers)
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
    def __init__(self, ch_scale=1, fil_sizes=(9, 5, 5)):
        super(SRCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, ch_scale * 32, ksize=fil_sizes[0], stride=1, pad=fil_sizes[0] // 2)
            self.conv2 = L.Convolution2D(
                None, ch_scale * 16, ksize=fil_sizes[1], stride=1, pad=fil_sizes[1] // 2)
            self.conv3 = L.Convolution2D(
                None, 1, ksize=fil_sizes[2], stride=1, pad=fil_sizes[2] // 2)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h


class DRLSR(chainer.Chain):
    def __init__(self):
        super(DRLSR, self).__init__()
        with self.init_scope():
            self.conv1_3 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1)
            self.conv1_5 = L.Convolution2D(None, 8, ksize=5, stride=1, pad=2)
            self.conv1_9 = L.Convolution2D(None, 8, ksize=9, stride=1, pad=4)
            self.conv2 = L.Convolution2D(None, 16, ksize=1, stride=1, pad=0)
            self.conv22 = L.Convolution2D(None, 16, ksize=3, stride=1, pad=1)
            self.conv23 = L.Convolution2D(None, 16, ksize=1, stride=1, pad=0)
            self.conv3_3 = L.Convolution2D(None, 8, ksize=3, stride=1, pad=1)
            self.conv3_5 = L.Convolution2D(None, 8, ksize=5, stride=1, pad=2)
            self.conv3_9 = L.Convolution2D(None, 8, ksize=9, stride=1, pad=4)
            self.conv4 = L.Convolution2D(None, 1, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        h = F.concat((F.relu(self.conv1_3(x)),
                      F.relu(self.conv1_5(x)),
                      F.relu(self.conv1_9(x))), axis=1)

        h = F.relu(self.conv2(h))
        h = F.relu(self.conv22(h))
        h = F.relu(self.conv23(h))

        h = F.concat((F.relu(self.conv3_3(h)),
                      F.relu(self.conv3_5(h)),
                      F.relu(self.conv3_9(h))), axis=1)

        h = F.relu(self.conv4(h))

        return h + x

class VDSR(chainer.Chain):
    def __init__(self, depth=5):
        super(VDSR, self).__init__()
        with self.init_scope():
            self.conv_in = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.conv_out = L.Convolution2D(None, 1, ksize=3, stride=1, pad=1)
            self._forward_list = []
            for i in range(depth - 2):
                name = 'conv_{}'.format(i + 1)
                conv = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
                setattr(self, name, conv)
                self._forward_list.append(name)

    def __call__(self, x):
        h = F.relu(self.conv_in(x))
        for name in self._forward_list:
            l = getattr(self, name)
            h = F.relu(l(h))
        h = F.relu(self.conv_out(h))
        return h + x


def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    img = Image.open(path.join(ROOT_PATH, 'dataset/Set_14/baboon.bmp')).convert('L')
    img = np.array(img)
    h, w = img.shape
    r_img = imresize(img, (h//3, w//3), interp='bicubic')
    img = imresize(r_img, (h, w), interp='bicubic')
    Image.fromarray(img).save('bic.bmp')
    img = img.reshape(1, 1, img.shape[0], img.shape[1]).astype(np.float32)
    img = img / 255.0


    # parameter出力

    # 保存ディレクトリ
    # save didrectory

   # prepare model
    model = GenEvaluator(SRCNN())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    serializers.load_npz(path.join(ROOT_PATH, 'models', 'SRCNN_Scl_1_Fil_955'), model)

    data = img

    data = chainer.cuda.to_gpu(data)
    y = data
    for i in range(10):
        y = model.generator(y)
        _y = chainer.cuda.to_cpu(y.data.reshape(h, w))
        _y = (_y * 255).astype(np.uint8)
        Image.fromarray(_y).save('{}_j.bmp'.format(i))
    Image.fromarray(r_img).save('tiisai.bmp')

if __name__ == '__main__':
    main()
