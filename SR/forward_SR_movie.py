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
from skimage import io, color, transform
from tqdm import tqdm
import cupy as cp

import networks as N

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

def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # データの読み込み
    data_path = path.join(ROOT_PATH, 'dataset', 'demo_videos', '09_Streets_of_India_960x540_s001_image')
    image_list = os.listdir(data_path)

    # データサイズ取得
    img = io.imread(path.join(data_path, image_list[0]))
    h, w, _ = img.shape
    # gray_images = np.zeros((len(image_list),1, h, w), dtype=np.float32)
    # cbcr_images = np.zeros((len(image_list),2, h, w), dtype=np.float32)

   # prepare model
    # model = GenEvaluator(N.SRCNN(2, [9, 5, 5]))
    model = GenEvaluator(N.DRLSR())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    serializers.load_npz(path.join(ROOT_PATH, 'models', 'drlsr'), model)

    for idx, image_name in tqdm(enumerate(image_list)):
        img = io.imread(path.join(data_path, image_name))
        img = transform.rescale(transform.rescale(img, 0.5), 2)
        io.imsave('/media/shimo/HDD_storage/BICUBIC/bic_' + image_name, img)

        img = color.rgb2yuv(img).astype(np.float32)

        img_gray = img[:, :, 0].reshape(1, 1, h, w)
        img_cbcr = img[:, :, 1:].transpose(2, 0, 1).reshape(1, 2, h, w)


        data = img_gray
        data = cp.array(data)
        y_data = chainer.backends.cuda.to_cpu(model.generator(data).data)
        img_gray = np.clip(y_data, 0.,1.)

        img = np.concatenate((img_gray, img_cbcr), axis=1)
        img = img.transpose(0, 2, 3, 1).reshape(h, w, 3)

        # print(img.dtype, img.shape, img.max(), img.min())
        # print(img[:,:,0].max(), img[:,:,0].min())
        # print(img[:,:,1].max(), img[:,:,1].min())
        # print(img[:,:,2].max(), img[:,:,2].min())
        img = np.clip(color.yuv2rgb(img) * 255, 0, 255).astype(np.uint8)
        # print(img.dtype, img.shape, img.max(), img.min())
        # print(img[:,:,0].max(), img[:,:,0].min())
        # print(img[:,:,1].max(), img[:,:,1].min())
        # print(img[:,:,2].max(), img[:,:,2].min())
        io.imsave('/media/shimo/HDD_storage/DRLSR_demo/drlsr_' + image_name, img.astype(np.uint8))
        # io.imsave('srcnn_gray_' + image_name, gray_images[i][0]/255.)


if __name__ == '__main__':
    main()
