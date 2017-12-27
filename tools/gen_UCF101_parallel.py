'''
UCF101のデータセットからsequenceデータを生成するスクリプト(修正版)
'''
import os
from os import path
import argparse
import random
import json
from glob import glob
import numpy as np
import pims
from tqdm import tqdm
from joblib import Parallel, delayed
import csv

# 実行ファイル位置
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))
#生データセットのパス
DATA_PATH = '/media/shimo/HDD_storage/DataSet/UCF-101'

def proc_act_movie(act, args, save_path, save_path_rel):
# 全動画ファイル取得
    act_path = path.join(DATA_PATH, act)
    npz_path_loc_list = []
    # 全グループ25の中からnum_ groupだけ選ぶ
    for gidx, group in enumerate(random.sample(range(1, 25 + 1), args.num_group)):
        # 該当するカットを選択
        movie_cuts = glob(path.join(act_path, 'v_{:}_g{:02d}'.format(act, group) + '_c*.avi'))
        movie_path = movie_cuts[random.randint(0, len(movie_cuts) - 1)]

        # ここから動画を切り刻む pimsでな
        movie = np.array(pims.Video(movie_path))
        # フレームとサイズの高さ幅の分割数
        f, h, w, ch = movie.shape
        h_split, w_split, f_split = h // args.size, w // args.size, f // args.frame

        # クリッピング
        movie = movie[: f_split * args.frame, :h_split * args.size, :w_split * args.size, :]

        # Patchサイズに分割
        movie = movie.reshape(f_split, args.frame, h_split, args.size, w_split, args.size, ch)
        # fsp, hsp, wsp, Nf, ch, H, Wに順番を入れ替え
        movie = movie.transpose((0, 2, 4, 1, 6, 3, 5))
        # データ数をプール
        movie = movie.reshape(-1, args.frame, ch, args.size, args.size).astype(np.float32) / 255.0

        x_idx = [i for i in range(args.frame) if i != args.frame // 2]
        #sequenceを保存するディレクトリ作成
        seq_save_path = path.join(save_path, act, 'group{}'.format(gidx))
        if not path.exists(seq_save_path):
            os.makedirs(seq_save_path)

        # sequenceをカウント
        count_seq = 0
        # 各sequenceに対して
        for seq in movie:
            # 分散が閾値以下なら無視
            if seq.var() < args.var_coef:
                continue
            # 特に理由もなくランダムに25%で無視
            if random.random() < 0.25:
                continue

            # データを保存する
            seq_name = path.join(seq_save_path, '{}_sequence_{}'.format(act, count_seq))
            x_data = seq[x_idx, :, :, :]
            y_data = seq[args.frame // 2, :, :, :]
            np.savez(seq_name, x_data=x_data, y_data=y_data)
            npz_path_loc_list.append(save_path_rel + '/{}/group{}/{}_sequence_{}.npz'.format(act, gidx, act, count_seq))
            count_seq += 1
            if count_seq >= args.max_sequence:
                break
    return npz_path_loc_list

def main():
    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=64,
                        help='size of sequence, width and height are same size.')
    parser.add_argument('--frame', '-f', type=int, default=3,
                        help="frames of sequence")
    parser.add_argument('--var_coef', '-v', type=float, default=0.003,
                        help='ignore sequence if frame-wise var less than this coef.')
    parser.add_argument('--num_group', '-g', type=int, default=10,
                        help='number of scene groups')
    parser.add_argument('--max_sequence', '-m', type=int, default=100,
                        help='max sequence')
    parser.add_argument('--name', '-n', type=str, default='train',
                        help='dataset name(e.g. train or test ...)')
    args = parser.parse_args()

    save_path_rel = 'UCF101_{}_size{}_frame{}_group{}_max{}_p'.format(args.name, args.size, args.frame, args.num_group, args.max_sequence)
    save_path = path.join(ROOT_PATH, 'dataset', save_path_rel)

    # parameter出力
    print("### DataSet Parameter ###")
    print("# Size of Sequence: {}".format(args.size))
    print("# Frame of Sequence: {}".format(args.frame))
    print("# Coef of Variance: {}".format(args.var_coef))
    print('# Number of Groups of Scenes: {}'.format(args.num_group))
    print('# Max Sequence: {}'.format(args.max_sequence))
    print("# Save Dir:", save_path)
    print("")

    # ディレクトリ作成
    if not path.exists(save_path):
        os.makedirs(save_path)

    # Parameter 保存
    with open(path.join(save_path, 'arg_params.json'), 'w') as f:
        json.dump(vars(args), f)

    # action名 取得
    act_list = os.listdir(DATA_PATH)

    # 各 Parallel計算
    print("Parallel計算")
    npz_path_loc_list = Parallel(n_jobs=6,verbose=10)(delayed(proc_act_movie)(act, args, save_path, save_path_rel) for act in act_list)
    npz_path_loc_list = [p for inner in npz_path_loc_list for p in inner]

    with open(path.join(save_path, 'train_data_loc.csv'), 'w') as f:
        for p in npz_path_loc_list:
            f.writelines(p + '\n')

if __name__ == '__main__':
    main()
