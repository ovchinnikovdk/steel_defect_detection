import os
import pandas as pd
import numpy as np


class DatasetGenerator:
    def __init__(self):
        pass

    def generate(self, path, test_split):
        df = pd.read_csv(path)
        # df = df[df['EncodedPixels'].notnull()].reset_index(drop=True)
        df['filename'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        df['class'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

        images = pd.DataFrame()
        images['filename'] = df['filename'].unique()
        images['rles'] = df.groupby(df.filename)
        images['rles'] = images['rles'].apply(lambda x: x[1][['class', 'EncodedPixels']].values)
        images['rles'] = images['rles'].apply(lambda x: list(map(lambda k: k[1], sorted(x, key=lambda val: val[0]))))
        test = images.sample(frac=test_split)
        train = images.drop(test.index)
        train['full'] = train['rles'].apply(lambda x: np.any([isinstance(t, str) for t in x]))
        train = train[train.full].reset_index()
        del train['full']
        return test, train
