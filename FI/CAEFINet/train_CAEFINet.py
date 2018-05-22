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
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

# 自作ネットワーク, データセット読み込み
import networks as N
import dataset as ds
import cupy as cp
#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../../'))

DATA_PATH = path.join(ROOT_PATH, 'dataset')

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
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU1 ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--iter_parallel', '-p', action='store_true', default=False,
                        help='loading dataset from disk')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test Mode, a few dataset')
    parser.add_argument('--opt' , '-o', type=str, choices=('adam', 'sgd') ,default='adam')
    parser.add_argument('--fsize' , '-f', type=int ,default=5)
    parser.add_argument('--ch' , '-c', type=int ,default=4)
    parser.add_argument('--decay' , '-d', type=str ,default='exp', choices=('exp', 'lin'))
    parser.add_argument('--weight', '-w', type=float ,default=1.0)
    args = parser.parse_args()

    # parameter出力
    print("-=Learning Parameter=-")
    print("# Max Epochs: {}".format(args.epoch))
    print("# Batch Size: {}".format(args.batchsize))
    print("# Learning Rate: {}".format(args.learnrate))
    print("# Optimizer Method: {}".format(args.opt))
    print("# Filter Size: {}".format(args.fsize))
    print("# Channel Scale: {}".format(args.ch))
    print("# coef. decay : {}".format(args.decay))
    print("# contloss' weight : {}".format(args.weight))
    print('# Train Dataet: General 100')
    if args.iter_parallel:
        print("# Data Iters that loads in Parallel")
    print("\n")

    # 保存ディレクトリ
    # save didrectory
    model_dir_name = 'CAEFINet_opt_{}_ch_{}_fsize_{}_decay_{}_weight_{}'.format(args.opt, args.ch, args.fsize, args.decay, args.weight)
    outdir = path.join(ROOT_PATH, 'results','FI' ,'CAEFINet', model_dir_name)
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

    #loading dataset
    if args.test:
        print('# loading test dataet(UCF101_minimam_test_size64_frame3_group2_max4_p) ...')
        train_dataset = 'UCF101_minimam_test_size64_frame3_group2_max4_p'
        test_dataset = 'UCF101_minimam_test_size64_frame3_group2_max4_p'
    else:
        print('# loading test dataet(UCF101_train_size64_frame3_group10_max100_p, UCF101_test_size64_frame3_group25_max5_p) ...')
        train_dataset = 'UCF101_train_size64_frame3_group10_max100_p'
        test_dataset = 'UCF101_test_size64_frame3_group25_max5_p'

    if args.iter_parallel:
        train = ds.SequenceDataset(dataset=train_dataset)
        test = ds.SequenceDataset(dataset=test_dataset)
    else:
        train = ds.SequenceDatasetOnMem(dataset=train_dataset)
        test = ds.SequenceDatasetOnMem(dataset=test_dataset)

   # prepare model
    model = N.CAEFINet(vgg_path=path.join(ROOT_PATH, 'models', 'VGG16.npz'), f_size=args.fsize, n_ch=args.ch, size=64)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

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
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, loss_func=model.get_loss_func(weight=args.weight, coef_decay=args.decay))
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # # eval test data
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.get_loss_func(weight=args.weight, coef_decay=args.decay)))
    # dump loss graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # lr shift
    if args.opt == 'sgd':
        trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=(50, 'epoch'))
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
    trainer.extend(
        extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                            'epoch', file_name='mse_loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/cont_loss', 'validation/main/cont_loss'],
                            'epoch', file_name='cont_loss.png'))
    # plot acc graph
    trainer.extend(extensions.PlotReport(['main/psnr', 'validation/main/psnr'],
                            'epoch', file_name='PSNR.png'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss','main/mse_loss', 'validation/main/mse_loss',
        'main/cont_loss', 'validation/main/cont_loss', 'main/psnr', 'validation/main/psnr', 'lr', 'elapsed_time']))
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
    model_name = 'CAEFINet_{}_ch_{}_fsize_{}_decay_{}_weight_{}.npz'.format(args.opt, args.ch, args.fsize, args.decay, args.weight)
    chainer.serializers.save_npz(path.join(model_outdir, model_name), model)

    model_parameter = {
        'name': 'CAEFINetConcat',
        'parameter': {'f_size':args.fsize, 'ch':args.ch}
    }
    with open(path.join(model_outdir, 'model_parameter.json'), 'w') as f:
        json.dump(model_parameter, f)

if __name__ == '__main__':
    main()
