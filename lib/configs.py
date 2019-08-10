import json
from lib.models import SimpleCNN, UNet
from lib import dataset
from lib import metrics
from torch.optim import Adam, SGD
import pandas as pd
import torch
import os
import tqdm


architectures = {
    'simplecnn': SimpleCNN,
    'unet': UNet
}

optimizers = {
    'adam': Adam,
    'sgd': SGD
}

losses = {
    'mse': torch.nn.MSELoss,
    'bcewithlogits': torch.nn.BCEWithLogitsLoss,
    'bce': torch.nn.BCELoss
}

datasets = {
    'stealdataset': dataset.StealDataset
}

metrics = {
    'dice': metrics.dice
}


class ConfigFactory:
    @staticmethod
    def build_model(json_path):
        with open(json_path, 'r') as json_file:
            conf = json.load(json_file)
            name = conf['model'].lower()
            del conf['model']
            return architectures[name](**conf)

    @staticmethod
    def build_train_env(net, json_path):
        with open(json_path, 'r') as json_file:
            conf = json.load(json_file)

            if 'batch_size' not in conf:
                conf['batch_size'] = 64
            if 'epochs' not in conf:
                conf['epochs'] = 30
            if 'test_split' not in conf:
                conf['test_split'] = 0.1
            print(conf)
            if conf['gpu']:
                conf['net'] = net.cuda()
            else:
                conf['net'] = net
            conf['loss'] = losses[conf['loss'].lower()]()
            conf['optimizer'] = optimizers[conf['optimizer']['class'].lower()](net.parameters(), **conf['optimizer']['params'])

            # Generating Datasets
            df = pd.read_csv(os.path.join(conf['dataset']['params']['base_path'], 'train.csv'))
            # df = df[df['EncodedPixels'].notnull()].reset_index(drop=True)
            df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
            df['class'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

            # TODO: Need to parallelize this block! Works slowly
            # UPD: Fixed
            # ----------------------------------------------
            images = pd.DataFrame()
            images['filename'] = df['filename'].unique()
            images['rles'] = df.groupby(df.filename)
            images['rles'] = images['rles'].apply(lambda x: x[1][['class', 'EncodedPixels']].values)
            images['rles'] = images['rles'].apply(lambda x: list(map(lambda k: k[1], sorted(x, key=lambda val: val[0]))))
            # for i in tqdm.tqdm(range(1, 5), desc='Preparing csv file'):
            #     images['class_' + str(i)] = images['filename']\
            #         .apply(lambda x: df[(df.filename == x) & (df['class'] == str(i))]['EncodedPixels'].values[0])
            test = images.sample(frac=conf['test_split'])
            train = images.drop(test.index)
            # ----------------------------------------------
            conf['dataset']['params']['df'] = train
            conf['train_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            conf['dataset']['params']['df'] = test
            conf['valid_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            del conf['dataset']
            del conf['test_split']

            # Metrics
            conf['metrics'] = {metric: metrics[metric] for metric in conf['metrics']}

            return conf


# if __name__ == '__main__':
#     configurer = ConfigFactory()
#     param_dir = os.path.join(os.pardir, 'params')
#     model_dir = os.path.join(param_dir, 'models')
#     train_dir = os.path.join(param_dir, 'trains')
#     model = configurer.build_model(os.path.join(model_dir, 'simple_cnn.json'))
#     train_params = configurer.build_train_env(model, os.path.join(train_dir, 'train_conf_1.json'))
#     print(train_params)
# python train.py --model_conf=./params/models/simple_cnn.json --train_conf=./params/trains/train_conf_1.json