from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from lib.mask_utils import rle2mask
from lib.transform import data_transform


class StealDataset(Dataset):
    def __init__(self, base_path, df, transform=data_transform, subset="train"):
        super().__init__()
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
        filename, classId = self.df['ImageId_ClassId'].iloc[index].split('_')
        fn = filename
        img = Image.open(self.data_path + fn)
        img = self.transform(img)

        if self.subset == 'train':
            mask = rle2mask(self.df['EncodedPixels'].iloc[index], (256, 1600))
            mask = transforms.ToPILImage()(mask)
            mask = self.transform(mask)
            return img, mask
        else:
            return img
