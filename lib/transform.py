from torchvision.transforms import Normalize
from torchvision import transforms
from albumentations import HorizontalFlip, VerticalFlip, Resize, Compose, RandomRotate90
from albumentations.torch import ToTensor

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
                RandomRotate90(),
                HorizontalFlip(),
                VerticalFlip()
            ]
        )
    list_transforms.extend(
        [
            Resize(64, 400),
            # Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
