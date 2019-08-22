from torchvision import transforms
from albumentations import HorizontalFlip, VerticalFlip, Resize, Compose, CoarseDropout
import albumentations as albu
from albumentations.torch import ToTensor
from lib.custom_crop import CustomCrop

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 400)),
    transforms.ToTensor()])

post_process = transforms.Compose([

])

data_clf_transform = transforms.Compose([
    transforms.Resize((128, 800)),
    transforms.ToTensor()])


def get_transforms(phase, mean=None, std=None):
    list_transforms = []
    if phase == "train":
        pass
    list_transforms.extend(
        [
            CustomCrop(256, 400),
            albu.CoarseDropout(),
            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
            albu.OneOf([
                HorizontalFlip(),
                VerticalFlip()
            ])
        ]
    )
    # if phase == 'val':
    #     list_transforms.extend(
    #         [
    #             Resize(128, 800)
    #         ]
    #     )
    list_transforms.extend(
        [
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


