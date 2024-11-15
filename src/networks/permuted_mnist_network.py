from .cbp_layer import CBPLinear
from .redo_layer import ReDoLinear

import torch

INPUT_DIMS = 784
OUTPUT_DIMS = 10


class ThreeHiddenLayerNetwork(torch.nn.Module):

    def __init__(self,
                 hidden_dim: int = 10,
                 use_skip_connections: bool = False,
                 preactivation_skip_connection: bool = False,
                 use_cbp=False,
                 maturity_threshold: int = None,
                 replacement_rate: float = None,
                 cbp_utility: str = "contribution",
                 use_redo=False,
                 reinit_frequency: int = None,
                 reinit_threshold: float = None,
                 redo_utility: str = "original",
                 use_layer_norm: bool = False,
                 preactivation_layer_norm: bool = False,
                 use_crelu: bool = False,
                 use_bottleneck: bool = False):
        """
        Three-layer ReLU network with continual backpropagation for MNIST
        """
        super().__init__()

        self.use_crelu = use_crelu
        input_dim_scaling = 1
        if self.use_crelu:
            assert hidden_dim % 2 == 0
            hidden_dim  = hidden_dim // 2
            input_dim_scaling = 2

        self.use_skip_connections = use_skip_connections
        self.preactivation_skip_connection = preactivation_skip_connection

        self.use_redo = use_redo
        self.use_cbp = use_cbp
        if self.use_redo and self.use_cbp:
            raise ValueError("Cannot use ReDo and CBP at the same time!")

        self.use_layer_norm = use_layer_norm
        self.preactivation_layer_norm = preactivation_layer_norm

        self.mt = maturity_threshold
        self.rr = replacement_rate
        self.rf = reinit_frequency
        self.rt = reinit_threshold

        # first layer
        self.ff_1 = torch.nn.Linear(INPUT_DIMS, out_features=hidden_dim, bias=True)
        self.act_1 = torch.nn.ReLU()
        self.neg_act_1 = torch.nn.ReLU()
        self.reinit_layer_1 = None          # either CBP or ReDo
        self.ln_1 = torch.nn.LayerNorm(hidden_dim * input_dim_scaling) if self.use_layer_norm else None
        # second layer
        second_layer_dim = hidden_dim if not use_bottleneck else hidden_dim // 10
        self.ff_2 = torch.nn.Linear(hidden_dim * input_dim_scaling, out_features=second_layer_dim, bias=True)
        self.act_2 = torch.nn.ReLU()
        self.neg_act_2 = torch.nn.ReLU()
        self.reinit_layer_2 = None
        self.ln_2 = torch.nn.LayerNorm(second_layer_dim * input_dim_scaling) if self.use_layer_norm else None
        # third layer
        self.ff_3 = torch.nn.Linear(second_layer_dim * input_dim_scaling, out_features=hidden_dim, bias=True)
        self.act_3 = torch.nn.ReLU()
        self.neg_act_3 = torch.nn.ReLU()
        self.reinit_layer_3 = None
        self.ln_3 = torch.nn.LayerNorm(hidden_dim * input_dim_scaling) if self.use_layer_norm else None
        self.out = torch.nn.Linear(hidden_dim * input_dim_scaling, OUTPUT_DIMS, bias=True)

        if use_cbp:
            assert maturity_threshold is not None and replacement_rate is not None
            self.reinit_layer_1 = CBPLinear(in_layer=self.ff_1, out_layer=self.ff_2, replacement_rate=self.rr,
                                            maturity_threshold=self.mt, ln_layer=self.ln_1, util_type=cbp_utility)
            self.reinit_layer_2 = CBPLinear(in_layer=self.ff_2, out_layer=self.ff_3, replacement_rate=self.rr,
                                            maturity_threshold=self.mt, ln_layer=self.ln_2, util_type=cbp_utility)
            self.reinit_layer_3 = CBPLinear(in_layer=self.ff_3, out_layer=self.out, replacement_rate=self.rr,
                                            maturity_threshold=self.mt, ln_layer=self.ln_3, util_type=cbp_utility)

        if self.use_redo:
            assert reinit_frequency is not None and reinit_threshold is not None
            self.reinit_layer_1 = ReDoLinear(in_layer=self.ff_1, out_layer=self.ff_2, reinit_frequency=self.rf,
                                             reinit_threshold=self.rt, ln_layer=self.ln_1, util_type=redo_utility)
            self.reinit_layer_2 = ReDoLinear(in_layer=self.ff_2, out_layer=self.ff_3, reinit_frequency=self.rf,
                                             reinit_threshold=self.rt, ln_layer=self.ln_2, util_type=redo_utility)
            self.reinit_layer_3 = ReDoLinear(in_layer=self.ff_3, out_layer=self.out, reinit_frequency=self.rf,
                                             reinit_threshold=self.rt, ln_layer=self.ln_3, util_type=redo_utility)

    def forward(self, x: torch.Tensor, activations: list = None) -> torch.Tensor:
        # first hidden layer
        x = self.ff_1(x)
        if self.use_layer_norm and self.preactivation_layer_norm:       # use layer norm before activations
            x = self.ln_1(x)
        x = torch.cat([self.act_1(x), self.neg_act_1(-x)], dim=-1) if self.use_crelu else self.act_1(x)
        if activations is not None: activations.append(x)               # store activations
        res = x                                                         # store residual connection
        if self.reinit_layer_1 is not None:                             # log features using cbp or redo
            x = self.reinit_layer_1(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:   # use layer norm after activation
            x = self.ln_1(x)

        # second hidden layer
        x = self.ff_2(x)
        if self.use_layer_norm and self.preactivation_layer_norm:
            x = self.ln_2(x)
        if self.use_skip_connections and self.preactivation_skip_connection:    # add residual connection before activation
            x = x + res
        x = torch.cat([self.act_2(x), self.neg_act_2(-x)], dim=-1) if self.use_crelu else self.act_2(x)
        if activations is not None: activations.append(x)
        if self.use_skip_connections and not self.preactivation_skip_connection:    # add residual connection after activation
            x = x + res
        res = x
        if self.reinit_layer_2 is not None:
            x = self.reinit_layer_2(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:
            x = self.ln_2(x)

        # third hidden layer
        x = self.ff_3(x)
        if self.use_layer_norm and self.preactivation_layer_norm:
            x = self.ln_3(x)
        if self.use_skip_connections and self.preactivation_skip_connection:
            x = x + res
        x = torch.cat([self.act_3(x), self.neg_act_3(-x)], dim=-1) if self.use_crelu else self.act_3(x)
        if activations is not None: activations.append(x)
        if self.use_skip_connections and not self.preactivation_skip_connection:
            x = x + res
        if self.reinit_layer_3 is not None:
            x = self.reinit_layer_3(x)
        if self.use_layer_norm and not self.preactivation_layer_norm:
            x = self.ln_3(x)

        return self.out(x)

    def feature_replace_event_indicator(self):
        if not self.use_cbp and not self.use_redo:
            return False

        return (self.reinit_layer_1.replace_feature_event_indicator or
                self.reinit_layer_2.replace_feature_event_indicator or
                self.reinit_layer_3.replace_feature_event_indicator)

    def reset_indicators(self):
        if not self.use_cbp and not self.use_redo: return

        self.reinit_layer_1.replace_feature_event_indicator = False
        self.reinit_layer_2.replace_feature_event_indicator = False
        self.reinit_layer_3.replace_feature_event_indicator = False
