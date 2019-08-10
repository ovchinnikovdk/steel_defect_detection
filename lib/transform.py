from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Scale((64, 400)),
    transforms.ToTensor()])
