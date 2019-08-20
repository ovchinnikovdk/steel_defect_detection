from torchvision import transforms
from albumentations import HorizontalFlip, VerticalFlip, Resize, Compose, ShiftScaleRotate, RandomCrop
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
        list_transforms.extend(
            [
                # ShiftScaleRotate(rotate_limit=3),
                CustomCrop(256, 400),
                HorizontalFlip(),
                VerticalFlip()
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
            # Resize(256, 256),
            # Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


