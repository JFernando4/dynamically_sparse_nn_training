from torch.nn import Conv2d, Linear, BatchNorm2d
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
from math import sqrt
import torch
import sys
from torch.nn.init import calculate_gain


def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


class ResGnT(object):
    """
    Generate-and-Test algorithm for a simple resnet, assuming only one fully connected layer at the top and that
    there is no pooling at the end
    """
    def __init__(self, net, hidden_activation, opt, decay_rate=0.99, replacement_rate=1e-4, init='kaiming',
                 util_type='weight', maturity_threshold=100, device='cpu', num_last_filter_outputs=1):
        super(ResGnT, self).__init__()

        self.net = net
        self.bn_layers = []
        self.weight_layers = []
        self.downsample_layers = []
        self.get_weight_layers(nn_module=self.net)
        self.num_hidden_layers = len(self.weight_layers) - 1

        self.opt = opt
        self.opt_type = 'sgd'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = torch.tensor(decay_rate, dtype=torch.float32, device=device)
        self.num_last_filter_outputs = num_last_filter_outputs
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util, self.ages, self.bias_corrected_util = [], [], []

        for i in range(self.num_hidden_layers):
            self.util.append(zeros(self.weight_layers[i].out_channels, dtype=torch.float32, device=device))
            self.ages.append(zeros(self.weight_layers[i].out_channels, dtype=torch.float32, device=device))
            self.bias_corrected_util.append(zeros(self.weight_layers[i].out_channels, dtype=torch.float32, device=device))

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)
        """
        Some pre-calculation
        """
        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                self.num_new_features_to_replace.append(self.replacement_rate * self.weight_layers[i].out_channels)

    def get_weight_layers(self, nn_module: torch.nn.Module):
        if isinstance(nn_module, Conv2d) or isinstance(nn_module, Linear):
            self.weight_layers.append(nn_module)
        elif isinstance(nn_module, BatchNorm2d):
            self.bn_layers.append(nn_module)
        else:
            for m in nn_module.children():
                if hasattr(nn_module, 'downsample'):
                    if nn_module.downsample == m:
                        self.downsample_layers.append(m)
                        continue
                self.get_weight_layers(nn_module=m)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        bounds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            # todo: fix layer bound calcualation for convlayers
            bounds.append(get_layer_bound(layer=self.weight_layers[i], init=init, gain=gain))
        bounds.append(get_layer_bound(layer=self.weight_layers[-1], init=init, gain=1))
        return bounds

    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            next_layer = self.weight_layers[layer_idx + 1]

            if isinstance(next_layer, Linear):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
            elif isinstance(next_layer, Conv2d):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))

            if self.util_type == 'weight':
                new_util = output_wight_mag
            elif self.util_type in ['contribution']:
                if isinstance(next_layer, Linear):
                    new_util = (output_wight_mag * features.abs().mean(dim=0)).view(-1, self.num_last_filter_outputs).mean(dim=1)
                elif isinstance(next_layer, Conv2d):
                    new_util = output_wight_mag * features.abs().mean(dim=(0, 2, 3))

            self.util[layer_idx] -=- (1 - self.decay_rate) * new_util
            # correct the bias in the utility computation
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace_input_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        features_to_replace_output_indices = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[i] -=- self.num_new_features_to_replace[i]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0: continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                           num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace_input_indices[i] = new_features_to_replace
            features_to_replace_output_indices[i] = new_features_to_replace
            if isinstance(self.weight_layers[i], Conv2d) and isinstance(self.weight_layers[i+1], Linear):
                features_to_replace_output_indices[i] = \
                    (new_features_to_replace*self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) + \
                    tensor([i for i in range(self.num_last_filter_outputs)], device=new_features_to_replace.device).repeat(new_features_to_replace.size()[0])

        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

    def update_optim_params(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Update Optimizer's state
        """
        pass

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer, next_layer = self.weight_layers[i], self.weight_layers[i+1]

                current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                current_layer.weight.data[features_to_replace_input_indices[i], :] += \
                    empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:])). \
                        uniform_(-self.bounds[i], self.bounds[i])

                current_layer.bias.data[features_to_replace_input_indices[i]] *= 0.0
                """
                Set the outgoing weights and ages to zero
                """
                next_layer.weight.data[:, features_to_replace_output_indices[i]] = 0
                self.ages[i][features_to_replace_input_indices[i]] = 0
                """
                Reset the corresponding batchnorm layers
                """
                self.bn_layers[i].bias[features_to_replace_input_indices[i]] *= 0.0
                self.bn_layers[i].weight[features_to_replace_input_indices[i]] *= 0.0
                self.bn_layers[i].weight[features_to_replace_input_indices[i]] += 1.0
                self.bn_layers[i].running_mean[features_to_replace_input_indices[i]] *= 0.0
                self.bn_layers[i].running_var[features_to_replace_input_indices[i]] *= 0.0
                self.bn_layers[i].running_var[features_to_replace_input_indices[i]] += 1.0

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
        self.update_optim_params(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
