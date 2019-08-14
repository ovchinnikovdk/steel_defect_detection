from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 400)),
    transforms.ToTensor()])

post_process = transforms.Compose([

])

data_clf_transform = transforms.Compose([
    transforms.Resize((128, 800)),
    transforms.ToTensor()])

