'''
動画データセットの作成スクリプト
動画データは事前に非圧縮の画像データに連番で切り分ける,
画像データの形式はtiffとする
'''
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

if __name__ == '__main__':
    #パス情報
    # images_path = '/media/shimo/HDD_storage/DataSet/SCENE_10/2K_SCENE/SCENE_01/*.tiff'
    images_path = '/media/shimo/HDD_storage/DataSet/SCENE_1/2K_images_mini/*.tiff'
    hdf5_path = '/home/shimo/CMULAB/STVSR/dataset/SCENE1_2K_mini.hdf5'
    #画像連番リスト
    image_names = sorted(glob.glob(images_path))
    #ソース動画のサイズ取得
    h_src, w_src, ch = cv2.imread(image_names[0]).shape
    del ch
    #ソース動画の画像枚数
    num_src = len(image_names)
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
    for i, name in enumerate(tqdm(image_names)):
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
    # x_data = buf.reshape(num_data, Nf, H, W).astype(np.float32) / 255
    buf = buf.reshape(num_data, Nf, H, W).astype(np.float32) / 255
    #データ作成
    y_data = buf[:, 1, :, :].reshape(num_data, 1, H, W)
    x_data = buf[:, [0, 2], :, :]
    #hdf5ファイル 書きこみ
    print('writing hdf5 file')
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('x_data', data=x_data, chunks=True)
        f.create_dataset('y_data', data=y_data, chunks=True)
        f.flush()

    print('finished')
