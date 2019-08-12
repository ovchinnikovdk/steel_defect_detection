from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from lib.mask_utils import rle2mask
from lib.transform import data_transform
import numpy as np
import torch
import cv2


class StealDataset(Dataset):
    def __init__(self, base_path, df, transform=data_transform, subset="train", size=None):
        super().__init__()
        if size is not None:
            self.df = df[:size]
        else:
            self.df = df
        self.transform = transform
        self.subset = subset

        if self.subset == "train":
            self.data_path = base_path + 'train_images/'
        elif self.subset == "test":
            self.data_path = base_path + 'test_images/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = self.df['filename'].iloc[index]
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == 'train':
            masks = []
            rles = self.df['rles'].iloc[index]
            assert len(rles) == 4, 'Need to be 4 classes for an image' + str(self.df['filename'].iloc[index])
            for i in range(len(rles)):
                mask = rle2mask(rles[i], (256, 1600))
                mask = cv2.resize(mask, (400, 64))
                masks.append(mask[None])
            mask = np.concatenate(masks, axis=0)
            # mask = transforms.ToPILImage()(mask)
            # mask = self.transform(mask)
            # mask = transforms.ToTensor()(mask)
            mask = torch.Tensor(mask)
            return img, mask
        else:
            return img, self.df['class'].iloc[index]
