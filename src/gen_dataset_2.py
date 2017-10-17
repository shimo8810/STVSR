'''
動画データセットの作成スクリプト
動画データは事前に非圧縮の画像データに連番で切り分ける,
画像データの形式はtiffとする
'''
import os
from os import path
import glob
import numpy as np
import h5py
import cv2
from tqdm import tqdm

#データセットパラメータ
#三次元データ (Nf, H, W) これ1つをsequenceと呼ぶ
#拡縮のスケールファクター3
W = 41
H = 41
Nf = 3
Scl = 3

# パス関連
DATA_PATH = '/media/shimo/HDD_storage/DataSet/SCENE_10/2K_SCENE'
SAVE_PATH = '/home/shimo/CMULAB/STVSR/dataset/SCENE10_2K_60fps.hdf5'

if __name__ == '__main__':
    #画像連番リスト
    outfile = h5py.File(SAVE_PATH, 'w')
    x_dset = outfile.create_dataset('x_data', shape=(1, 2, H, W), dtype=np.float32,
         maxshape=(None, 2, H, W))
    y_dset = outfile.create_dataset('y_data', shape=(1, 1, H, W), dtype=np.float32,
         maxshape=(None, 1, H, W))
    data_size = 0
    for scene_dir in sorted(os.listdir(DATA_PATH)):
        print(scene_dir)
        images_path = sorted(glob.glob(path.join(DATA_PATH, scene_dir, '*.tiff')))
        #ソース動画のサイズ取得
        h_src, w_src, _ = cv2.imread(images_path[0]).shape
        #ソース動画の画像枚数
        num_src = len(images_path)
        #画像の縦横の分割数
        h_sep, w_sep = h_src // H, w_src // W
        #フレームの分割数
        f_sep = num_src // Nf
        #最終的なデータ数(ピクセル分散を考慮しない場合)
        num_data = f_sep * h_sep * w_sep
        #データに関する情報出力
        print('source moive(or images):')
        print('width:{}\theight:{}\tframe:{}'.format(w_src, h_src, num_src))
        print('dataset parameter:')
        print('width:{}\theight:{}\tframe:{} \tnum:{} \t scale:{}'.format(
            W, H, Nf, num_data, Scl))

        #画像を読み込む
        images = np.zeros((h_src, w_src, num_src), dtype=np.uint8)
        print('reading movie(or images)')
        for i, name in enumerate(tqdm(images_path)):
            images[:, :, i] = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

        #画像を当分割できるサイズにクリッピング
        print('creating dataset')
        images = images[0:h_sep * H, 0:w_sep * W, 0:f_sep * Nf]

        #data作成
        #画像をPatchサイズに切り出し
        buf = images.reshape(h_sep, H, w_sep, W, f_sep, Nf)
        #h_sep, H, w_sep, W, f_sep, Nf -> #f_sep, h_sep, w_sep, Nf, H, W の並びに変換
        buf = buf.transpose((4, 0, 2, 5, 1, 3))
        #f_sep, h_sep, w_sep, Nf, H, W -> num_data, Nf, H, W の並びに変換
        buf = buf.reshape(num_data, Nf, H, W).astype(np.float32) / 255
        #データ作成
        y_data = buf[:, 1, :, :].reshape(num_data, 1, H, W)
        x_data = buf[:, [0, 2], :, :]
        #hdf5ファイル 書きこみ
        print(x_data.shape, x_data.dtype)
        print('writing hdf5 file')
        x_dset.resize((data_size + num_data, 2, H, W))
        y_dset.resize((data_size + num_data, 1, H, W))
        x_dset[data_size:data_size + num_data, :, :, :] = x_data
        y_dset[data_size:data_size + num_data, :, :, :] = y_data
        outfile.flush()
        data_size = data_size + num_data
        # del images, buf, x_data, y_data

    outfile.close()
    print('finished')
