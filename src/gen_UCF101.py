'''
動画データをnpzの配列データに変換するプログラム
'''
import os
from os import path
import glob
import numpy as np
import h5py
import cv2
from tqdm import tqdm
import ffmpy
#データセットパラメータ
#三次元データ (Nf, H, W) これ1つをsequenceと呼ぶ
#拡縮のスケールファクター3
W = 41
H = 41
Nf = 3
Scl = 3

# パス関連
DATA_PATH = '/media/shimo/HDD_storage/DataSet/UCF-101'
SAVE_PATH = '/media/shimo/HDD_storage/DataSet/UCF-101_Images'

if __name__ == '__main__':
    #全シーン取得
    ucf101_scenes = os.listdir(DATA_PATH)
    for scene in ucf101_scenes:
        print("#Reading movie in", scene)
        #動画リスト取得
        mv_list = os.listdir(path.join(DATA_PATH, scene))
        for movie in mv_list:
            # 動画を読み込み連番画像へ変換
            name = movie.split('.')[0]
            print(movie, name)
            inputs = path.join(DATA_PATH, scene, movie)
            outputs = path.join(SAVE_PATH, 'tmp', '{}%04d.tiff'.format(name))
            ff = ffmpy.FFmpeg(
                inputs={inputs: None},
                outputs={outputs: '-f image2'}
            )
            ff.run()

            # 切り出した連番画像をnumpyで全て読み込み
            img_list = os.listdir(path.join(SAVE_PATH, 'tmp'))
            buf = cv2.imread(path.join(SAVE_PATH, 'tmp', img_list[0]), cv2.IMREAD_GRAYSCALE)
            img_set = np.zeros((len(img_list), *buf.shape))
            for i, img in enumerate(sorted(img_list)):
                img_set[i, :, :] = cv2.imread(path.join(SAVE_PATH, 'tmp', img), cv2.IMREAD_GRAYSCALE)
            #データセット保存
            print("#Saving data file ...")
            np.save(path.join(SAVE_PATH, name), img_set)
            print("#Remove img file ...")
            for i in range(len(img_list)):
                os.remove(path.join(SAVE_PATH, 'tmp', img_list[i]))

