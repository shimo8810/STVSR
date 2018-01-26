import os
from os import path
import json
import chainer
#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

# DATA_PATH = '/media/shimo/HDD_storage/DataSet'
DATA_PATH = path.join(ROOT_PATH, 'dataset')

def make_result_dir(args, outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))

def save_trained_model(model_name, model, net_name, **kwargs):
    # save final model
    model_outdir = path.join(ROOT_PATH, 'models', model_name)
    if not path.exists(model_outdir):
        os.makedirs(model_outdir)
    model_save_name = '{}.npz'.format(model_name)
    chainer.serializers.save_npz(path.join(model_outdir, model_save_name), model)

    model_parameter = {
        'name': net_name,
        'parameter': kwargs,
    }
    with open(path.join(model_outdir, 'model_parameter.json'), 'w') as f:
        json.dump(model_parameter, f)
