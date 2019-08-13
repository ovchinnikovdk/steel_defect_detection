import torch
import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from lib.mask_utils import pred2mask, mask2rle
import cv2


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
            predicts = []
            self.model.eval()
            for data, _ in tqdm.tqdm(self.loader, desc='Predicting: CleanSteelPredictor'):
                if self.cuda:
                    data = data.cuda()
                output = self.model(data)
                predicts += output.cpu()[0]
            self.df['hasDefect'] = predicts
            clean = self.df[self.df['hasDefect'] < 0.4]
            self.df = self.df.drop(clean.index)
            clean['EncodedPixels'] = ' '
            del self.df['hasDefect']
            del clean['hasDefect']
            return clean, self.df


class SteelSegmentation(object):
    def __init__(self, df, model, loader, cuda):
        self.df = df
        self.model = model
        self.loader = loader
        self.cuda = cuda

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
                output = self.model(data)
                output = output.cpu()
                masks = [pred2mask(mask[lab - 1]).numpy() for mask, lab in zip(output, label)]
                rles = Parallel(n_jobs=mp.cpu_count())(delayed(self.post_process)(img) for img in masks)
                predicts += rles
            self.df['EncodedPixels'] = predicts
            return self.df

    def post_process(self, img):
        resized = cv2.resize(img, (1600, 256))
        return mask2rle(resized)
