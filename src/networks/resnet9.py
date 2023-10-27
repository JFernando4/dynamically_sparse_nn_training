import torch


def kaiming_init_resnet_module(nn_module: torch.nn.Module):
    """
    Initializes the parameters of a resnet module in the following way:
        - Conv2d: weights are initialized using xavier normal initialization and bias are initialized to zero
        - Linear: same as Conv2d
        - BatchNorm2d: bias are initialized to 0, weights are initialized to 1
    :param nn_module: an instance ot torch.nn.Module to be initialized
    """

    if isinstance(nn_module, torch.nn.Conv2d) or isinstance(nn_module, torch.nn.Linear):
        if isinstance(nn_module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="relu")
        else:   # the only linear layer in a resnet is the output layer
            torch.nn.init.kaiming_normal_(nn_module.weight, nonlinearity="linear")
        if nn_module.bias is not None:
            torch.nn.init.constant_(nn_module.bias, 0.0)

    if isinstance(nn_module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(nn_module.weight, 1.0)
        torch.nn.init.constant_(nn_module.bias, 0.0)

"""
The implementation bellow is from this website:
    https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min#Classifying-CIFAR10-images-using-a-ResNet-and-Regularization-techniques-in-PyTorch
"""

def conv_block(in_channels, out_channels, pool=False, norm_function=torch.nn.BatchNorm2d):
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        norm_function(out_channels),
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
