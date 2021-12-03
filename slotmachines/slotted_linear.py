from math import sqrt

import torch

import torch.nn.functional as F
from .selection_engine import GreedySelection, ProbabilisticSelection


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, k=8, greedy_selection=True, bias=True, device=None, dtype=None):
        super(Linear, self).__init__()
        self.k = k

        # the weights retain their initial values through training
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features, self.k), device=device, dtype=dtype, requires_grad=False))
        self.score = torch.nn.Parameter(torch.empty((out_features, in_features, self.k), device=device, dtype=dtype))

        if bias:

            self.bias = torch.nn.Parameter(torch.empty((out_features, 1), device=device, dtype=dtype, requires_grad=False))
            self.bias_score = torch.nn.Parameter(torch.empty((out_features, self.k), device=device, dtype=dtype))
        else:

            # equivalent to self.bias = None plus some other stuff
            self.register_parameter('bias', None)

        # IMPORTANT: this initialization is probably overwritten by the transformer
        self._xavier_uniform_()

        if greedy_selection:
            self._selection_engine = GreedySelection.apply
        else:
            self._selection_engine = ProbabilisticSelection.apply

    def forward(self, x):
        selected_net = self._selection_engine(self.score)
        net = torch.sum(self.weight * selected_net, dim=-1)

        if self.bias is None:

            out = F.linear(x, net)
        else:

            selected_bias = self._selection_engine(self.bias_score)
            bias = torch.sum(self.bias * selected_bias, dim=-1)
            out = F.linear(x, net, bias)

        return out

    # their implementation is somewhat different from PyTorch and Tim's initialization
    def _xavier_uniform_(self):
        w, h, _ = self.weight.size()
        # +1 for bias
        std = sqrt(6.0 / float(w + h + 1)) if self.bias is not None else sqrt(6.0 / float(w + h)) 
        self.weight.data.uniform_(-std, std)
        self.score.data.uniform_(0, std)

        if self.bias is not None:

            self.bias.data.uniform_(-std, std)
            self.bias_score.data.uniform_(0, std)
