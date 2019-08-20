import json
from lib.models import *
from lib import dataset
from lib import metrics
from lib.preprocessing import *
from torch.optim import Adam, SGD
from torchvision import models
import pandas as pd
import numpy as np
import torch
import os
import tqdm


architectures = {
    'unet': UNet,
    'deeplabv3plus': DeepLabV3Plus,
    'resnet': models.ResNet,
    'densenet': DenseNetSigmoid,
    'pspnet': PSPNet
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
    'steeldataset': dataset.SteelDataset,
    'steeldatasetv2': dataset.SteelDatasetV2,
    'steelpredictiondataset': dataset.SteelPredictionDataset
}

metrics = {
    'dice': metrics.dice,
    'rocauc': metrics.roc_auc,
    'accuracy': metrics.accuracy_score
}

generators = {
    'segmentationdatasetgenerator': SegmentationDatasetGenerator,
    'classifierdatasetgenerator': ClassifierDatasetGenerator
}


class ConfigFactory:
    def build_model(self, json_path):
        with open(json_path, 'r') as json_file:
            conf = json.load(json_file)
            name = conf['model'].lower()
            del conf['model']
            return architectures[name](**conf)

    def build_train_env(self, net, json_path):
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
            test, train = generators[conf['generator'].lower()]()\
                .generate(os.path.join(conf['dataset']['params']['base_path'], 'train.csv'), conf['test_split'])
            # ----------------------------------------------
            conf['dataset']['params']['df'] = train
            conf['train_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            conf['dataset']['params']['df'] = test
            conf['dataset']['params']['subset'] = 'val'
            conf['valid_data'] = datasets[conf['dataset']['class'].lower()](**conf['dataset']['params'])
            del conf['dataset']
            del conf['test_split']
            del conf['generator']

            # Metrics
            conf['metrics'] = {metric: metrics[metric] for metric in conf['metrics']}
            conf['net_version'] = json_path.split('/')[-1].split('.')[0]

            return conf

    def build_submit_env(self, json_path):
        with open(json_path) as json_file:
            config = json.load(json_file)
            df = pd.read_csv(config['csv'])
            if 'size' in config:
                df = df.sample(config['size'])
            cuda = config['cuda']
            models = dict()
            # df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
            # df['class'] = df['ImageId_ClassId'].apply(lambda x: int(x.split('_')[1]))
            if 'segmentation' in config:
                seg_model = self.build_model(config['segmentation']['model'])
                seg_model.load_state_dict(torch.load(config['segmentation']['state_dict']))
                models['segmentation'] = seg_model
            if 'prediction' in config:
                pred_model = self.build_model(config['prediction']['model'])
                pred_model.load_state_dict(torch.load(config['prediction']['state_dict']))
                models['prediction'] = pred_model
            return models, df, config['data_path'], cuda



# if __name__ == '__main__':
#     configurer = ConfigFactory()
#     param_dir = os.path.join(os.pardir, 'params')
#     model_dir = os.path.join(param_dir, 'models')
#     train_dir = os.path.join(param_dir, 'trains')
#     model = configurer.build_model(os.path.join(model_dir, 'unet1.json'))
#     train_params = configurer.build_train_env(model, os.path.join(train_dir, 'train_unet_local1.json'))
#     print(train_params)
# python train.py --model_conf=./params/models/unet1.json --train_conf=./params/trains/train_local_1.json