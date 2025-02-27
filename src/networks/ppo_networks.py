"""
This code is adapted from the loss-of-plasticity repository for the paper: Loss of plasticity in deep continual learning
Link to the github repo: https://github.com/shibhansh/loss-of-plasticity/tree/main
"""
from torch import nn
import torch
from torch.distributions import Normal

from .cbp_layer import CBPLinear
from .redo_layer import ReDoLinear
from .shifted_layer_norm import ShiftedLayerNorm


def register_hook(net: nn.Module, hook_fn):

    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            register_hook(layer, hook_fn)
        else:
            # it's a non-sequential. Register a hook
            layer.register_forward_hook(hook_fn)


class TwoLayerNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 act_type="ReLU",
                 h_dim: int = 256,
                 use_cbp: bool = False,
                 maturity_threshold: int = 0,
                 replacement_rate: float = 0.0,
                 use_redo: bool = False,
                 reinit_frequency: int = 0,
                 reinit_threshold: float = 0.0,
                 decay_rate: float = 0.99,
                 use_ln: bool = True,
                 use_shifted_ln: bool = False):
        """
        Two-hidden-layer network with continual backpropagation and ReDo for PPO experiments
        """
        super().__init__()
        self.use_cbp = use_cbp
        self.use_redo = use_redo
        self.act_type = act_type
        self.activation_func = {'Tanh': nn.Tanh, 'ReLU': nn.ReLU, 'elu': nn.ELU, 'sigmoid': nn.Sigmoid}[self.act_type]
        self.use_ln = use_ln
        layer_norm_class = ShiftedLayerNorm if use_shifted_ln else nn.LayerNorm

        # first layer
        self.ff_1 = nn.Linear(input_dim, h_dim)
        self.act_1 = self.activation_func()
        self.reinit_layer_1 = None
        self.ln_1 = layer_norm_class(h_dim) if self.use_ln else None
        self.weights_per_feature_1 = input_dim + h_dim

        # second layer
        self.ff_2 = nn.Linear(h_dim, h_dim)
        self.act_2 = self.activation_func()
        self.reinit_layer_2 = None
        self.ln_2 = layer_norm_class(h_dim) if self.use_ln else None
        self.weights_per_feature_2 = h_dim + output_dim

        # output layer
        self.out = nn.Linear(h_dim, output_dim)

        if self.use_cbp:
            assert maturity_threshold > 0 and replacement_rate > 0.0
            kw_arguments = {"maturity_threshold": maturity_threshold, "replacement_rate": replacement_rate,
                            "util_type": "contribution", "decay_rate": decay_rate, "use_shifted_ln": use_shifted_ln}
            self.reinit_layer_1 = CBPLinear(in_layer=self.ff_1, out_layer=self.ff_2, ln_layer=self.ln_1, **kw_arguments)
            self.reinit_layer_2 = CBPLinear(in_layer=self.ff_2, out_layer=self.out, ln_layer=self.ln_2, **kw_arguments)

        if self.use_redo:
            assert reinit_frequency > 0 and reinit_threshold > 0.0
            kw_arguments = {"reinit_frequency": reinit_frequency, "reinit_threshold": reinit_threshold,
                            "util_type": "original", "decay_rate": decay_rate, "use_shifted_ln": use_shifted_ln}
            self.reinit_layer_1 = ReDoLinear(in_layer=self.ff_1, out_layer=self.ff_2, ln_layer=self.ln_1, **kw_arguments)
            self.reinit_layer_2 = ReDoLinear(in_layer=self.ff_2, out_layer=self.out, ln_layer=self.ln_2, **kw_arguments)

    def forward(self, x: torch.Tensor, activations: list = None):

        # first hidden layer
        x = self.ff_1(x)
        x = self.act_1(x)
        if activations is not None:
            activations.append(x)
        if self.reinit_layer_1 is not None:
            x = self.reinit_layer_1(x)
        if self.use_ln:
            x = self.ln_1(x)

        # second hidden layer
        x = self.ff_2(x)
        x = self.act_2(x)
        if activations is not None:
            activations.append(x)
        if self.reinit_layer_2 is not None:
            x = self.reinit_layer_2(x)
        if self.use_ln:
            x = self.ln_2(x)

        # output layer
        x = self.out(x)

        return x

    def feature_replace_event_indicator(self):
        if not self.use_cbp and not self.use_redo: return False

        return (self.reinit_layer_1.replace_feature_event_indicator or
                self.reinit_layer_2.replace_feature_event_indicator)

    def reset_indicators(self):
        if not self.use_cbp and not self.use_redo: return

        self.reinit_layer_1.reset_indicators()
        self.reinit_layer_2.reset_indicators()

    def num_replaced(self):
        if not self.use_redo and not self.use_cbp: return (0, 0, 0)
        nr_1 = 0.0 if not self.reinit_layer_1.replace_feature_event_indicator else self.reinit_layer_1.num_replaced
        nr_2 = 0.0 if not self.reinit_layer_2.replace_feature_event_indicator else self.reinit_layer_2.num_replaced
        return nr_1 * self.weights_per_feature_1, nr_2 * self.weights_per_feature_2


def initialize_two_layer_network(net: TwoLayerNetwork, use_shifted_ln: bool = False):
    """ Initializes network using kaiming uniform distribution """
    act = "relu" if net.act_type == "elu" else net.act_type.lower()

    linear_layers = (net.ff_1, net.ff_2)
    ln_layers = (net.ln_1, net.ln_2)

    for layer in linear_layers:
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity=act)
        torch.nn.init.zeros_(layer.bias)

    for layer in ln_layers:
        if layer is not None:
            if use_shifted_ln:
                torch.nn.init.zeros_(layer.weight)
            else:
                torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    torch.nn.init.zeros_(net.out.weight)
    torch.nn.init.zeros_(net.out.bias)


class MLPVF(nn.Module):
    def __init__(self, input_dim,
                 act_type='ReLU',
                 h_dim: int = 256,
                 device='cpu',
                 use_cbp: bool = False,
                 maturity_threshold: int = 0,
                 replacement_rate: float = 0.0,
                 use_redo: bool = False,
                 reinit_frequency: int = 0,
                 reinit_threshold: float = 0.0,
                 decay_rate: float = 0.99,
                 use_ln: bool = True,
                 use_shifted_ln: bool = False):

        super().__init__()

        # setup network architecture
        self.v_net = TwoLayerNetwork(input_dim=input_dim, output_dim=1, h_dim=h_dim, act_type=act_type,
                                     use_cbp=use_cbp, maturity_threshold=maturity_threshold, replacement_rate=replacement_rate,
                                     use_redo=use_redo, reinit_frequency=reinit_frequency, reinit_threshold=reinit_threshold,
                                     decay_rate=decay_rate, use_ln=use_ln, use_shifted_ln=use_shifted_ln)
        self.v_net.to(device)

    def value(self, x, activations: list = None):
        val = self.v_net.forward(x, activations)
        return val


class MLPPolicy(nn.Module):
    def __init__(self,
                 input_dim: int,
                 a_dim: int,
                 act_type: str = 'ReLU',
                 h_dim: int = 256,
                 log_std=0,
                 device='cpu',
                 use_cbp: bool = False,
                 maturity_threshold: int = 0,
                 replacement_rate: float = 0.0,
                 use_redo: bool = False,
                 reinit_frequency: int = 0,
                 reinit_threshold: float = 0.0,
                 decay_rate: float = 0.99,
                 use_ln: bool = True,
                 use_shifted_ln: bool = False):
        super().__init__()

        self.act_type = act_type
        self.device = device

        self.mean_net = TwoLayerNetwork(input_dim=input_dim, output_dim=a_dim, h_dim=h_dim, act_type=act_type,
                                        use_cbp=use_cbp, maturity_threshold=maturity_threshold, replacement_rate=replacement_rate,
                                        use_redo=use_redo, reinit_frequency=reinit_frequency, reinit_threshold=reinit_threshold,
                                        decay_rate=decay_rate, use_ln=use_ln, use_shifted_ln=use_shifted_ln)

        self.log_std = nn.Parameter(torch.ones(a_dim) * log_std)
        self.mean_net.to(device)
        self.discrete_actions = False

    def action(self, x, activations: list = None):
        """
        :param x: tensor of shape [N, 1], where N is number of observations
        :return:
            action: of shape [N, 1]
            lprob: of shape [N, 1]
        """
        with torch.no_grad():
            dist = self.dist(x, activations)
            action = dist.sample()
            lprob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, lprob, dist

    def logp_dist(self, x, a, activations: list = None):
        dist = self.dist(x, activations)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device)).sum(1, keepdim=True)
        return lprob, dist

    def dist(self, x, activations: list = None):
        x = x.to(self.device)
        action_mean = self.mean_net.forward(x, activations)
        return Normal(action_mean, torch.exp(self.log_std))

    def dist_to(self, dist: Normal, to_device='cpu'):
        dist.loc.to(to_device)
        dist.scale.to(to_device)
        return dist

    def dist_stack(self, dists, device='cpu'):
        return Normal(
            torch.cat(tuple([dists[i].loc for i in range(len(dists))])).to(device),
            torch.cat(tuple([dists[i].scale for i in range(len(dists))])).to(device)
        )

    def dist_index(self, dist: Normal, ind):
        return Normal(dist.loc[ind], dist.scale[ind])