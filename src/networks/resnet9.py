import torch


def conv_block(in_channels, out_channels, pool=False, norm_function=torch.nn.BatchNorm2d):
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)


class ResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_classes, norm_function=torch.nn.BatchNorm2d):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, norm_function=norm_function)
        self.conv2 = conv_block(64, 128, pool=True, norm_function=norm_function)
        self.res1 = torch.nn.Sequential(
            conv_block(128, 128, norm_function=norm_function), conv_block(128, 128, norm_function=norm_function))

        self.conv3 = conv_block(128, 256, pool=True, norm_function=norm_function)
        self.conv4 = conv_block(256, 512, pool=True, norm_function=norm_function)
        self.res2 = torch.nn.Sequential(
            conv_block(512, 512, norm_function=norm_function), conv_block(512, 512, norm_function=norm_function))

        self.classifier = torch.nn.Sequential(torch.nn.MaxPool2d(4),
                                              torch.nn.Flatten(),
                                              torch.nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
