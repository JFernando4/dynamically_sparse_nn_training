import torch


class SequentialWithKeywordArguments(torch.nn.Sequential):

    """
    Sequential module that allows the use of keyword arguments in the forward pass
    """

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input
