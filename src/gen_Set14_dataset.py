'''
静止画像データをnumpy配列データに変換
データaugはchainer側でやるのでこっちは特に何もしないよ
'''
import os
from os import path
from PIL import Image
import numpy as np
from scipy.misc import imresize
import h5py
from tqdm import tqdm

# 実行ファイル位置
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))
#データセットのパス
DATA_PATH = '/media/shimo/HDD_storage/DataSet/Set_14'
# 保存先のパス
SAVE_PATH = '/media/shimo/HDD_storage/DataSet/super_resolution_HDF5_dataset'

#parameter
SIZE_INPUT = 64
SIZE_LABEL = 64
HEIGHT, WIDTH = 41, 41
SCALE = 3
STRIDE = 23

if __name__ == '__main__':
    #全画像ファイル名取得
    image_list = sorted(os.listdir(DATA_PATH))
    h5file = h5py.File(path.join(SAVE_PATH, 'Set14_test.hdf5'), 'w')

    x_dset = h5file.create_dataset('x_data',
                                   shape=(1, HEIGHT, WIDTH), dtype=np.float32, maxshape=(None, HEIGHT, WIDTH))
    y_dset = h5file.create_dataset('y_data',
                                   shape=(1, HEIGHT, WIDTH), dtype=np.float32, maxshape=(None, HEIGHT, WIDTH))

    num_data = 0
    for name in tqdm(image_list):
        # print(name)
        img = np.array(Image.open(path.join(DATA_PATH, name)).convert('L'))
        # print(img.shape)
        img_h, img_w = img.shape

        for y in range(0, img_h - HEIGHT, STRIDE):
            for x in range(0, img_w - WIDTH, STRIDE):
                crop_img = img[y:y + HEIGHT, x:x + WIDTH]
                if (crop_img.shape) != (HEIGHT, WIDTH):
                    continue

                dwn_img = imresize(
                    crop_img, (HEIGHT // SCALE, WIDTH // SCALE), interp='bicubic')
                dwn_img = imresize(dwn_img, (HEIGHT, WIDTH), interp='bicubic')
                x_dset.resize((num_data + 1, HEIGHT, WIDTH))
                y_dset.resize((num_data + 1, HEIGHT, WIDTH))
                x_dset[num_data, :, :] = crop_img
                y_dset[num_data, :, :] = dwn_img
                num_data += 1
                h5file.flush()

    h5file.close()
