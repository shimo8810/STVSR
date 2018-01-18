'''
train SRCNN Network
simple network
'''
import os
from os import path
import argparse
import random
import csv
from tqdm import tqdm
import platform
import json

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
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

# 自作ネットワーク, データセット読み込み
import networks as N
import datasets as ds
import cupy as cp
#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

# DATA_PATH = '/media/shimo/HDD_storage/DataSet'
DATA_PATH = path.join(ROOT_PATH, 'dataset')

def transform_sequence(x_data, y_data):
    '''
    sequence データのデータAUGを行う
    '''
    # 水平方向に反転するかどうか
    if random.choice([True, False]):
        x_data = np.flip(x_data, 2)
        y_data = np.flip(y_data, 1)

    # 垂直方向に反転するかどうか
    if random.choice([True, False]):
        x_data = np.flip(x_data, 3)
        y_data = np.flip(y_data, 2)

    # 90度回転するかどうか
    if random.choice([True, False]):
        x_data = np.rot90(x_data, axes=(2, 3))
        y_data = np.rot90(y_data, axes=(1, 2))

    return cp.array(x_data), cp.array(y_data)

class SequenceDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='train'):
        self.image_paths = []
        csv_path = None
        if dataset == 'train':
            csv_path = 'UCF101_train_size64_frame5_group10_max64_p/train_data_loc.csv'
        elif  dataset == 'test':
            csv_path = 'UCF101_test_size64_frame5_group25_max4_p/train_data_loc.csv'

        with open(path.join(DATA_PATH, csv_path)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.image_paths.append(path.join(DATA_PATH, row[0]))

    def __len__(self):
        return len(self.image_paths)

    def get_example(self, i):
        data = np.load(self.image_paths[i])
        x_data = data['x_data']
        y_data = data['y_data']
        return transform_sequence(x_data, y_data)
        # return x_data, y_data

def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.001,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', type=int, default=0,
                        help='GPU1 ID (negative value indicates CPU)')
    parser.add_argument('--gpu1', '-G', type=int, default=2,
                        help='GPU2 ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--iter_parallel', '-p', action='store_true', default=False,
                        help='loading dataset from disk')
    parser.add_argument('--opt' , '-o', type=str, choices=('adam', 'sgd') ,default='adam')
    parser.add_argument('--fsize' , '-f', type=int ,default=5)
    parser.add_argument('--ch' , '-c', type=int ,default=4)
    args = parser.parse_args()

    # parameter出力
    print("-=Learning Parameter=-")
    print("# Max Epochs: {}".format(args.epoch))
    print("# Batch Size: {}".format(args.batchsize))
    print("# Learning Rate: {}".format(args.learnrate))
    print("# Optimizer Method: {}".format(args.opt))
    print("# Filter Size: {}".format(args.fsize))
    print("# Channel Scale: {}".format(args.ch))
    print('# Train Dataet: General 100')
    if args.iter_parallel:
        print("# Data Iters that loads in Parallel")
    print("\n")

    # 保存ディレクトリ
    # save didrectory
    model_dir_name = 'AEFINet_concat_2_parallel_opt_{}_ch_{}_fsize_{}'.format(args.opt, args.ch, args.fsize)
    outdir = path.join(ROOT_PATH, 'results','FI' ,'AEFINet', model_dir_name)
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    #loading dataset
    print('# loading dataet(General100_train, General100_test) ...')
    if args.iter_parallel:
        train = ds.SequenceDataset(dataset='train')
        test = ds.SequenceDataset(dataset='test')
    else:
        train = ds.SequenceDatasetOnMem(dataset='train')
        test = ds.SequenceDatasetOnMem(dataset='test')

    chainer.cuda.get_device_from_id(args.gpu0).use()

   # prepare model
    model = N.GenEvaluator(N.AEFINetConcat2(f_size=args.fsize, ch=args.ch))
        # model.to_gpu()

    # setup optimizer
    if args.opt == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=args.learnrate)
    elif args.opt == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # setup iter
    if args.iter_parallel:
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=8)
        test_iter = chainer.iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False, n_processes=8)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test, args.batchsize, repeat=False, shuffle=False)


    # setup trainer
    updater = training.ParallelUpdater(
        train_iter,
        optimizer,
        devices={'main': args.gpu0, 'second': args.gpu1},
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # # eval test data
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu0))
    # dump loss graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # lr shift
    if args.opt == 'sgd':
        trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=(100, 'epoch'))
    elif args.opt == 'adam':
        trainer.extend(extensions.ExponentialShift("alpha", 0.1), trigger=(50, 'epoch'))
    # save snapshot
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_snapshot_{.updater.epoch}'), trigger=(10, 'epoch'))
    # log report
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    #  plot loss graph
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))
    # plot acc graph
    trainer.extend(
        extensions.PlotReport(
            ['main/PSNR', 'validation/main/PSNR'],
            'epoch', file_name='PSNR.png'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/PSNR', 'validation/main/PSNR', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    # [ChainerUI] enable to send commands from ChainerUI
    trainer.extend(CommandsExtension())
    # [ChainerUI] save 'args' to show experimental conditions
    save_args(args, outdir)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # save final model
    model_outdir = path.join(ROOT_PATH, 'models', model_dir_name)
    if not path.exists(model_outdir):
        os.makedirs(model_outdir)
    model_name = 'AEFINet_concat_2_opt_{}_ch_{}_fsize_{}.npz'.format(args.opt, args.ch, args.fsize)
    chainer.serializers.save_npz(path.join(model_outdir, model_name), model)

    model_parameter = {
        'name': 'AEFINetConcat2',
        'parameter': {'f_size':args.fsize, 'ch':args.ch}
    }
    with open(path.join(model_outdir, 'model_parameter.json'), 'w') as f:
        json.dump(model_parameter, f)

if __name__ == '__main__':
    main()
