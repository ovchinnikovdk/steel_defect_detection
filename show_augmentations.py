from lib.dataset import SteelDatasetV2
from lib.preprocessing import SegmentationDatasetGenerator
import cv2
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip, VerticalFlip, Compose, ShiftScaleRotate
from lib.custom_crop import CustomCrop

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
                ShiftScaleRotate(rotate_limit=3),
                CustomCrop(256, 400),
                HorizontalFlip(),
                VerticalFlip()
            ]
        )
    list_trfms = Compose(list_transforms)
    return list_trfms


path = './input/severstal-steel-defect-detection/'
test, train = SegmentationDatasetGenerator().generate(path + 'train.csv', 0.1)
train_dataset = SteelDatasetV2(path, train, transforms_func=example_transforms, size=100)

for i in range(len(train_dataset)):
    img, mask = train_dataset[i]
    # img = img.reshape(img.shape[0], img.shape[1])
    save_mask_image('img' + str(i), img, mask)
