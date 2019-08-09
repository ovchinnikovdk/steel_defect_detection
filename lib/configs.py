import json
from lib.models import SimpleCNN
from lib import dataset
from torch.optim import Adam
import pandas as pd
import torch
import os


architectures = {
    'simplecnn': SimpleCNN
}

optimizers = {
    'adam': Adam
}

losses = {
    'mse': torch.nn.MSELoss
}

datasets = {
    'stealdataset': dataset.StealDataset
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
            df = df[df['EncodedPixels'].notnull()].reset_index(drop=True)
            test = df.sample(frac=conf['test_split'])
            train = df.drop(test.index)
            conf['dataset']['params']['df'] = train
            conf['train_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            conf['dataset']['params']['df'] = test
            conf['valid_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            del conf['dataset']
            del conf['test_split']

            # Metrics
            conf['metrics'] = {'EER': lambda x, y: 1.0}

            return conf


# if __name__ == '__main__':
#     configurer = ConfigFactory()
#     param_dir = os.path.join(os.pardir, 'params')
#     model_dir = os.path.join(param_dir, 'models')
#     train_dir = os.path.join(param_dir, 'trains')
#     model = configurer.build_model(os.path.join(model_dir, 'simple_cnn.json'))
#     train_params = configurer.build_train_env(model, os.path.join(train_dir, 'train_conf_1.json'))
#     print(train_params)
