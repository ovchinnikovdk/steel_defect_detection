import torch
import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from lib.mask_utils import pred2mask, mask2rle
import cv2
import numpy as np


class CleanSteelPredictor(object):
    def __init__(self, df, model, dataloader, cuda):
        self.model = model
        self.loader = dataloader
        self.df = df
        self.cuda = cuda

    def call(self):
        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            predicts = np.array([])
            self.model.eval()
            for data in tqdm.tqdm(self.loader, desc='Predicting: CleanSteelPredictor'):
                if self.cuda:
                    data = data.cuda()
                output = self.model(data)
                predicts = np.concatenate((predicts, output.view(-1).cpu().numpy()))
            self.df['hasDefect'] = predicts
            clean = self.df[self.df['hasDefect'] < 0.3]
            print("{0:d} of {1:d} are clean".format(len(clean), len(self.df)))
            self.df = self.df.drop(clean.index)
            clean = clean.reset_index()
            self.df = self.df.reset_index()
            clean['EncodedPixels'] = clean['EncodedPixels'].apply(lambda x: ' ')

            return clean[['ImageId_ClassId', 'EncodedPixels']], self.df[['ImageId_ClassId', 'EncodedPixels']]


class SteelSegmentation(object):
    def __init__(self, df, model, loader, cuda, threshold=0.55):
        self.df = df
        self.model = model
        self.loader = loader
        self.cuda = cuda
        self.threshold = threshold

    def call(self):
        if self.cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        with torch.no_grad():
            predicts = []
            self.model.eval()
            for data, label in tqdm.tqdm(self.loader, desc='Predicting: SteelSegmentation'):
                if self.cuda:
                    data = data.cuda()
                if hasattr(self.model, 'predict'):
                    output = self.model.predict(data)
                else:
                    output = self.model(data)
                output = output.cpu()
                masks = [pred2mask(mask[lab - 1], threshold=self.threshold).numpy() for mask, lab in zip(output, label)]
                rles = Parallel(n_jobs=mp.cpu_count())(delayed(self.post_process)(img) for img in masks)
                predicts += rles
            self.df['EncodedPixels'] = predicts
            return self.df[['ImageId_ClassId', 'EncodedPixels']]

    def post_process(self, img):
        resized = cv2.resize(img, (1600, 256))
        return mask2rle(resized)
