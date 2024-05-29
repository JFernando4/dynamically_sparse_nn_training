from .cbp_layer import CBPLinear

import torch

INPUT_DIMS = 784
OUTPUT_DIMS = 10


class ThreeHiddenLayerNetwork(torch.nn.Module):

    def __init__(self,
                 hidden_dim: int = 10,
                 use_cbp=False,
                 maturity_threshold: int = None,
                 replacement_rate: float = None):
        """
        Three-layer ReLU network with continual backpropagation for MNIST
        """
        super().__init__()

        self.use_cbp = use_cbp
        self.mt = maturity_threshold
        self.rr = replacement_rate

        self.ff_1 = torch.nn.Linear(INPUT_DIMS, out_features=hidden_dim, bias=True)
        self.act_1 = torch.nn.ReLU()
        self.cbp_1 = None
        self.ff_2 = torch.nn.Linear(hidden_dim, out_features=hidden_dim, bias=True)
        self.act_2 = torch.nn.ReLU()
        self.cbp_2 = None
        self.ff_3 = torch.nn.Linear(hidden_dim, out_features=hidden_dim, bias=True)
        self.act_3 = torch.nn.ReLU()
        self.cbp_3 = None
        self.out = torch.nn.Linear(hidden_dim, OUTPUT_DIMS, bias=True)

        if use_cbp:
            assert maturity_threshold is not None and replacement_rate is not None
            self.cbp_1 = CBPLinear(in_layer=self.ff_1, out_layer=self.ff_2, replacement_rate=self.rr,
                                   maturity_threshold=self.mt, init="kaiming")
            self.cbp_2 = CBPLinear(in_layer=self.ff_2, out_layer=self.ff_3, replacement_rate=self.rr,
                                   maturity_threshold=self.mt, init="kaiming")
            self.cbp_3 = CBPLinear(in_layer=self.ff_3, out_layer=self.out, replacement_rate=self.rr,
                                   maturity_threshold=self.mt, init="kaiming")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # first hidden layer
        x = self.ff_1(x)
        x = self.act_1(x)
        if self.cbp_1 is not None:
            x = self.cbp_1(x)
        # second hidden layer
        x = self.ff_2(x)
        x = self.act_2(x)
        if self.cbp_2 is not None:
            x = self.cbp_2(x)
        # third hidden layer
        x = self.ff_3(x)
        x = self.act_3(x)
        if self.cbp_3 is not None:
            x = self.cbp_3(x)
        return self.out(x)

    def feature_replace_event_indicator(self):
        if not self.use_cbp:
            return False
        return (self.cbp_1.replace_feature_event_indicator or
                self.cbp_2.replace_feature_event_indicator or
                self.cbp_3.replace_feature_event_indicator)

    def reset_indicators(self):
        if not self.use_cbp: return
        self.cbp_1.replace_feature_event_indicator = False
        self.cbp_2.replace_feature_event_indicator = False
        self.cbp_3.replace_feature_event_indicator = False

    @torch.no_grad()
    def get_average_gradient_magnitude(self):
        """ Returns the average magnitude of the gradient of the parameters in the network """
        assert self.ff_1.weight.grad is not None
        parameter_list = [self.ff_1.weight, self.ff_1.bias, self.ff_2.weight, self.ff_2.bias,
                          self.ff_3.weight, self.ff_3.bias, self.out.weight, self.out.bias]
        parameter_count = 0
        gradient_magnitude_sum = 0.0
        for p in parameter_list:
            parameter_count += p.numel()
            gradient_magnitude_sum += p.grad.abs().sum()
        return gradient_magnitude_sum / parameter_count

