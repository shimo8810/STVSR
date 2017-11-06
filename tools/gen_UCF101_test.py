'''
動画データをnpyのnumpy配列データに変換し
そのファイル位置を示すデータファイルを生成するプログラム
'''
import os
from os import path
import random
import csv
import glob
import numpy as np
import cv2
import ffmpy

# 実行ファイル位置
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))
#データセットのパス
DATA_PATH = '/media/shimo/HDD_storage/DataSet/UCF-101'
# 保存先のパス
SAVE_PATH = '/media/shimo/HDD_storage/DataSet/Test_Mini_UCF101'

# ハイパラ
WIDTH = 41
HEIGHT = 41
NUM_FRAME = 3
VAR_COEF = 0.003
NUM_GRP = 10
MAX_SEQ = 10


if __name__ == '__main__':
    # アクション名 取得
    act_list = os.listdir(DATA_PATH)

    # データが保存されているパスを記述するためのCSVファイル作成
    csvfile = open(path.join(SAVE_PATH, 'train_data.csv'), 'w')
    csvfile_loc = open(path.join(SAVE_PATH, 'train_data_loc.csv'), 'w')
    # 101このアクションに対して
    for act in act_list:
        # 全動画ファイル名取得
        act_path = path.join(DATA_PATH, act)
        movie_list = os.listdir(act_path)

        for gidx, group in enumerate(random.sample(range(1, 25 + 1), NUM_GRP)):
            # 動画名 生成
            movie_name = 'v_{:}_g{:02d}_c01.avi'.format(act, group)

            if not movie_name in movie_list:
                raise ValueError('{} is not in {}'.format(movie_name, group))

            # 各動画を切り刻む ffmpegによってな
            save_name = '{}_f%05d.tiff'.format(movie_name.split('.')[0])
            ff = ffmpy.FFmpeg(
                inputs={path.join(act_path, movie_name): None},
                outputs={path.join(SAVE_PATH, 'tmp', save_name): '-f image2'}
            )
            ff.run()

            # 保存用ディレクトリの作成とか

            # 画像群データ処理部分
            tmp_img_name = '{}_f*.tiff'.format(movie_name.split('.')[0])
            img_path_list = glob.glob(
                path.join(SAVE_PATH, 'tmp', tmp_img_name))
            img_height, img_width, img_ch = cv2.imread(img_path_list[0]).shape
            images = np.zeros((len(img_path_list), img_height,
                               img_width, img_ch), dtype=np.uint8)

            # parameter計算
            # フレームと縦横の分割数
            h_split, w_split = img_height // HEIGHT, img_width // WIDTH
            f_split = len(images) // NUM_FRAME

            #全データ読み込み
            for idx, img in enumerate(img_path_list):
                images[idx, :, :, :] = cv2.imread(img)

            # フレームをクリッピング
            images = images[:f_split * NUM_FRAME,
                            : h_split * HEIGHT, : w_split * WIDTH, :]
            # Patchサイズに切り分け
            buf = images.reshape(f_split, NUM_FRAME, h_split,
                                 HEIGHT, w_split, WIDTH, img_ch)
            # fsp, hsp, wsp, Nf, ch, H, Wに順番を入れ替え
            buf = buf.transpose((0, 2, 4, 1, 6, 3, 5))
            # データ数をプール
            buf = buf.reshape(-1, NUM_FRAME, img_ch, HEIGHT, WIDTH) / 255.0

            # フレーム を分けるヤツ
            fidx = [i for i in range(NUM_FRAME) if i != NUM_FRAME // 2]

            # ディレクトリ作成とか
            # if not path.exists(path.join(SAVE_PATH, ))
            save_dir = path.join(SAVE_PATH, act, 'group{}'.format(gidx))
            if not path.exists(save_dir):
                os.makedirs(save_dir)
            seq_name = path.join(save_dir)
            # 各Sequenceに対して分散とか計算して導出
            count = 0
            for sequence in buf:
                # 分散が一定以下なら無視
                if sequence.var() < VAR_COEF:
                    continue
                if random.random() < 0.4:
                    continue
                seq_name = path.join(
                    save_dir, '{}_sequence_{}'.format(act, count))
                x_data = sequence[fidx, :, :, :].astype(np.float32)
                y_data = sequence[NUM_FRAME // 2, :, :,
                                  :].reshape(img_ch, HEIGHT, WIDTH).astype(np.float32)
                np.savez(seq_name, x_data=x_data, y_data=y_data)
                csvfile.writelines(seq_name + '.npz\n')
                csvfile_loc.writelines(
                    'Test_Mini_UCF101/{}/group{}/{}_sequence_{}.npz\n'.format(act, gidx, act, count))
                count += 1
                if count >= MAX_SEQ:
                    break
                # break

            # tmp内一掃
            for im in img_path_list:
                os.remove(im)

        #     break
        # break

    csvfile.close()
