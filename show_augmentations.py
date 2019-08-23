from lib.dataset import SteelDatasetV2
from lib.preprocessing import SegmentationDatasetGenerator
import cv2
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip, VerticalFlip, Compose, ShiftScaleRotate
from lib.custom_crop import CustomCrop
import albumentations as albu
import tqdm

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]


def save_mask_image(name, img, mask):
    fig, ax = plt.subplots(figsize=(15, 15))
    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.savefig('./logs/augmentations/' + name + '.png')


def example_transforms(phase, mean=None, std=None):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.IAAAdditiveGaussianNoise(p=0.1),
                albu.IAAPerspective(p=0.1, scale=(0.001, 0.005)),
                CustomCrop(256, 800),
                albu.CoarseDropout(max_holes=30,
                                   min_holes=5,
                                   max_height=5,
                                   max_width=5,
                                   min_height=3,
                                   min_width=3,
                                   p=0.2),
                albu.OneOf(
                    [
                        albu.CLAHE(p=1),
                        albu.RandomBrightnessContrast(p=1),
                        albu.RandomGamma(p=1),
                    ],
                    p=0.8,
                ),
                albu.OneOf(
                    [
                        albu.RandomBrightnessContrast(p=1),
                        albu.HueSaturationValue(p=1),
                    ],
                    p=0.8,
                ),
                albu.OneOf(
                    [
                        HorizontalFlip(p=1),
                        VerticalFlip(p=1)
                    ],
                    p=0.8
                )
            ]
        )
    list_trfms = Compose(list_transforms)
    return list_trfms


path = './input/severstal-steel-defect-detection/'
test, train = SegmentationDatasetGenerator().generate(path + 'train.csv', 0.1)
train_dataset = SteelDatasetV2(path, train, transforms_func=example_transforms, size=100)

for i in tqdm.tqdm(range(len(train_dataset))):
    img, mask = train_dataset[i]
    # img = img.reshape(img.shape[0], img.shape[1])
    save_mask_image('img' + str(i), img, mask)
