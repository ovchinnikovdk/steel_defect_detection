from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from lib.mask_utils import rle2mask
from lib.transform import *
import numpy as np
import torch
import cv2


class SteelDataset(Dataset):
    def __init__(self, base_path, df, transform=data_transform, subset="train", size=None,
                 original_size=(256, 1600), resize_to=(64, 400)):
        super().__init__()
        if size is not None:
            self.df = df[:size]
        else:
            self.df = df
        self.original_size = original_size
        self.resize_to = resize_to
        self.transform = transform
        self.subset = subset

        if self.subset == "train" or self.subset == "val":
            self.data_path = base_path + 'train_images/'
        elif self.subset == "test":
            self.data_path = base_path + 'test_images/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = self.df['filename'].iloc[index]
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == "train" or self.subset == "val":
            masks = []
            rles = self.df['rles'].iloc[index]
            assert len(rles) == 4, 'Need to be 4 classes for an image' + str(self.df['filename'].iloc[index])
            for i in range(len(rles)):
                mask = rle2mask(rles[i], (1600, 256))
                mask = cv2.resize(mask, (self.resize_to[1], self.resize_to[0]))
                masks.append(mask[None])
            mask = np.concatenate(masks, axis=0)
            # mask = transforms.ToPILImage()(mask)
            # mask = self.transform(mask)
            # mask = transforms.ToTensor()(mask)
            mask = torch.Tensor(mask)
            return img, mask
        else:
            return img, self.df['class'].iloc[index]


class SteelPredictionDataset(Dataset):
    def __init__(self, base_path, df, transform=data_clf_transform, subset='train', size=None):
        if size is not None:
            self.df = df.sample(size)
        else:
            self.df = df
        self.subset = subset
        self.transform = transform
        if self.subset == "train":
            self.data_path = base_path + 'train_images/'
        elif self.subset == "test":
            self.data_path = base_path + 'test_images/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = self.df['filename'].iloc[index].split('_')[0]
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == 'train':
            label = self.df['label'].iloc[index]
            return img, torch.Tensor(np.array([label]))
        else:
            return img


class SteelDatasetV2(Dataset):
    def __init__(self, base_path, df, transform=data_transform, subset="train", size=None,
                 original_size=(256, 1600), resize_to=(64, 400)):
        super().__init__()
        if size is not None:
            self.df = df[:size]
        else:
            self.df = df
        self.original_size = original_size
        self.resize_to = resize_to
        self.transform = get_transforms(subset)
        self.subset = subset

        if self.subset == "train" or self.subset == "val":
            self.data_path = base_path + 'train_images/'
        elif self.subset == "test":
            self.data_path = base_path + 'test_images/'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = self.df['filename'].iloc[index]
        img = cv2.imread(self.data_path + fn, 0)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        if self.subset == 'train' or self.subset == 'val':
            rles = self.df['rles'].iloc[index]
            assert len(rles) == 4, 'Need to be 4 classes for an image' + str(self.df['filename'].iloc[index])
            masks = np.zeros((256, 1600, 4), dtype=np.int32)
            for i in range(len(rles)):
                mask = rle2mask(rles[i], (1600, 256))
                masks[:, :, i] = mask
            # mask = np.concatenate(masks, axis=0)
            # Mask shape (256, 1600, 4)
            augmented = self.transform(image=img, mask=masks)
            img, mask = augmented['image'], augmented['mask']
            mask = mask[0].permute(2, 0, 1)
            return img, mask
        else:
            return self.transform(img), self.df['class'].iloc[index]