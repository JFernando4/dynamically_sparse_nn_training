
import torch


def inject_noise(net: torch.nn.Module, noise_std: float):
    """
    Adds a small amount of random noise to the parameters of the network. This is used for shrink-and-perturb.
    """

    with torch.no_grad():
        for param in net.parameters():
            param.add_(torch.randn(param.size(), device=param.device) * noise_std)
