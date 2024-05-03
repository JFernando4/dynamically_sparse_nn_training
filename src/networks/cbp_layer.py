import torch
from torch import nn
from math import sqrt


def call_reinit(m, i, o):
    m.reinit()


def log_features(m, i, o):
    with torch.no_grad():
        if m.decay_rate == 0:
            m.features = i[0]
        else:
            if m.features is None:
                m.features = (1 - m.decay_rate) * i[0]
            else:
                m.features = m.features * m.decay_rate + (1 - m.decay_rate) * i[0]


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nn.Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nn.Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


class CBPLinear(nn.Module):
    def __init__(
            self,
            in_layer: nn.Module,
            out_layer: nn.Module,
            act_type='relu',
            replacement_rate=0,
            init='kaiming',
            maturity_threshold=1000,
            util_type='contribution',
            decay_rate=0,
    ):
        super().__init__()
        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None

        """
        Register hooks
        """
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        # todo: add warning that in_layer and out_layers must be nn.Linear
        self.in_layer = in_layer
        self.out_layer = out_layer
        """
        Utility of all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.accumulated_num_features_to_replace = 0
        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace

        num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    return features_to_replace

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Calculate feature utility
        """
        output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)
        self.util.data = output_weight_mag * self.features.abs().mean(dim=[i for i in range(self.features.ndim - 1)])
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace = new_features_to_replace
        return features_to_replace

    def reinit_features(self, features_to_replace):
        """
        Reset input and output weights for low utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace.shape[0]
            if num_features_to_replace == 0: return
            self.in_layer.weight.data[features_to_replace, :] *= 0.0
            self.in_layer.weight.data[features_to_replace, :] += \
                torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).uniform_(-self.bound, self.bound)
            self.in_layer.bias.data[features_to_replace] *= 0

            self.out_layer.weight.data[:, features_to_replace] = 0
            self.ages[features_to_replace] = 0

    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)
        # todo: think if I should implement update optim parameters