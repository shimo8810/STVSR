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
from glob import glob

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

    # 保存先のディレクトリ作成
    result_path = path.join(ROOT_PATH, 'examples/result_Set14', args.model)
    if not path.exists(result_path):
        os.makedirs(result_path)

    img_path = glob(path.join(ROOT_PATH, 'examples/SR_example/Set_14/*.bmp'))
    for ip in img_path:
        # 画像読み込み
        img_name = ip.split('/')[-1].split('.')[0]
        img = io.imread(ip)

        h, w = img.shape[0], img.shape[1]

        # 拡縮挟んでbic
        img_low = imresize(img, (h // 3, w // 3), interp='bicubic')
        img_low = imresize(img_low, (h, w), interp='bicubic')

        img_yuv = color.rgb2yuv(img_low).astype(np.float32)
        img_y = img_yuv[:,:,0]

        # モデル読み込み
        model = N.GenEvaluator(N.SRCNN(ch_scale=2, fil_sizes=(9,5,5)))

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()
            import cupy as cp
            img_y = cp.array(img_y)

        # モデルファイル読み込み
        chainer.serializers.load_npz(path.join(ROOT_PATH, 'models/SRCNN_scl_2_fil_955_adam'), model)

        img_y = img_y.reshape(1, 1, img_y.shape[0], img_y.shape[1])

        print(img_y.max(), img_y.min(), img_y.mean(), img_y.dtype, img_y.shape)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                img_hr = model.generator(img_y).data # * 255
        img_hr = chainer.cuda.to_cpu(img_hr) * 255
        img_hr = img_hr.astype(np.uint8).reshape(img_y.shape[2], img_y.shape[3])
        print(img_hr.max(), img_hr.min(), img_hr.mean(), img_hr.dtype, img_hr.shape)

        # 画像を保存
        io.imsave(path.join(result_path,img_name + '_gray.bmp'), img_hr)

        print(img_yuv.dtype)
        buf = np.concatenate((img_hr.reshape(img_y.shape[2], img_y.shape[3], 1).astype(np.float32) / 255 , img_yuv[:,:,1:]), axis=2)
        buf = np.clip(color.yuv2rgb(buf) * 255, 0, 255).astype(np.uint8)
        io.imsave(path.join(result_path, img_name + '_color.bmp'), buf)

if __name__ == '__main__':
    main()
