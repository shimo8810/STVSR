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
from skimage import io
import cupy as cp
from tqdm import tqdm

import networks as N

# このファイルのパス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

FULL_PATH = '/media/shimo/HDD_storage/DataSet/harmonic_4K_videos/FHD_SIZE/demo'

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
    parser.add_argument('--model', '-m', type=str,
                        help=('Using Model'))
    args = parser.parse_args()

    # parameter出力

    # 保存ディレクトリ
    # save didrectory

   # prepare model
   # 個々でネットワークを帰る
    model = N.GenEvaluator(N.AEFINetConcat(ch=4, f_size=5))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model_path = path.join(ROOT_PATH, 'models', args.model)
    chainer.serializers.load_npz(model_path, model)

    movie_img = 'acrobat_s01'

    img_list = sorted(os.listdir(path.join(FULL_PATH, movie_img)))
    h, w, c = io.imread(path.join(FULL_PATH, movie_img, img_list[0]))[:, : ,:].shape
    h = 521
    w = 941

    # 保存用ディレクトリの作成
    save_path = path.join(ROOT_PATH, 'examples', 'result_acrobat_s01', args.model.split('.')[0])
    if not path.exists(save_path):
        os.makedirs(save_path)

    for i in tqdm(range(0,len(img_list)-1, 2)):
        img1_path = path.join(FULL_PATH, movie_img, img_list[i])
        img2_path = path.join(FULL_PATH, movie_img, img_list[i + 2])
        data = np.zeros((2, h, w, c)).astype(np.uint8)
        data[0] = io.imread(img1_path)[:h,:w,:]
        data[1] = io.imread(img2_path)[:h,:w,:]
        data = data.transpose(0, 3, 1, 2).reshape(1, 2, c, h, w).astype(np.float32) / 255.0
        data = cp.array(data)
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                data = model.generator(data)
        data = chainer.cuda.to_cpu(data.data) * 255
        data = np.clip(data, 0, 255)
        data = data.astype(np.uint8)
        data = data.reshape(c, h, w).transpose(1,2,0)
        name,_ = img_list[i].split('.')
        img1 = io.imread(img1_path)[:h,:w,:]
        io.imsave(path.join(save_path, '{:}_fi_{:04d}.bmp'.format(movie_img, i + 1)), img1)
        io.imsave(path.join(save_path, '{:}_fi_{:04d}.bmp'.format(movie_img, i + 2)), data)

if __name__ == '__main__':
    main()
