'''
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
from skimage import io, color
from scipy.misc import imresize

import networks as N
#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))


def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='Number of images in each mini-batch')
    parser.add_argument('--model', '-m', type=str,help=('Using Model'))
    args = parser.parse_args()

    img = io.imread(path.join(ROOT_PATH, 'examples/SR_example/Set_14/baboon.bmp'))

    img = img[:41, :41,:]

    img_low = imresize(img, (img.shape[0] // 2, img.shape[1] // 2), interp='bicubic')
    img_low = imresize(img_low, (img.shape[0], img.shape[1]), interp='bicubic')
    print(img_low.shape)

    img_yuv = color.rgb2yuv(img_low).astype(np.float32)
    print(img_yuv.shape, img_yuv.dtype, img_yuv[:,:,0].max(), img_yuv[:,:,0].min())
    img_y = img_yuv[:,:,0]
    io.imsave('bicubic.bmp', np.clip(img_y * 255, 0, 255).astype(np.uint8))

    print(img_y.shape, img_y.max(), img_y.min())

    # モデル読み込み
    model = N.GenEvaluator(N.SRCNN(ch_scale=2))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        import cupy as cp
        img_y = cp.array(img_y)

    # モデルファイル読み込み
    chainer.serializers.load_npz(path.join(ROOT_PATH, 'models/SRCNN_Scl_2_fil_955_adam'), model)

    img_y = img_y.reshape(1, 1, img_y.shape[0], img_y.shape[1])
    with chainer.using_config('train', False):
        with chainer.using_config('enable_backprop', False):
            img_hr = model.generator(img_y).data # * 255
    img_hr = chainer.cuda.to_cpu(img_hr) * 255
    img_hr = img_hr.astype(np.uint8).reshape(img_y.shape[2], img_y.shape[3])
    print(img_hr.max(), img_hr.min(), img_hr.mean(), img_hr.dtype, img_hr.shape)

    # 画像を保存
    io.imsave('hoge_sgd.bmp', img_hr)

    print(img_yuv.dtype)
    buf = np.concatenate((img_hr.reshape(img_y.shape[2], img_y.shape[3], 1).astype(np.float32) / 255 , img_yuv[:,:,1:]), axis=2)
    buf = np.clip(color.yuv2rgb(buf) * 255, 0, 255).astype(np.uint8)
    io.imsave('color.bmp', buf)

if __name__ == '__main__':
    main()
