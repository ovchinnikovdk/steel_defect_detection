import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=3):
        super(SimpleCNN, self).__init__()
        self.convs = torch.nn.Sequential()
        last_n_channels = in_channels
        for i in range(num_layers):
            self.convs.add_module('conv_block_' + str(i), torch.nn.Sequential(
                torch.nn.Conv2d(last_n_channels, last_n_channels + 5, kernel_size=3, padding=1),
                torch.nn.ReLU()
            ))
            last_n_channels += 5
        for i in range(num_layers - 1):
            self.convs.add_module('dconv_block_' + str(i), torch.nn.Sequential(
                torch.nn.Conv2d(last_n_channels, last_n_channels - 5, kernel_size=3, padding=1),
                torch.nn.ReLU()
            ))
            last_n_channels -= 5
        self.convs.add_module('last_conv', torch.nn.Sequential(
            torch.nn.Conv2d(last_n_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU()
        ))

    def forward(self, x):
        return self.convs(x)
