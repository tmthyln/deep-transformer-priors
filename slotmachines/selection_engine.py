from torch import autograd, eye, argmax, multinomial
from torch.nn import functional as F


class GreedySelection(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        """Select the weight corresponding to the highest score"""
        idx = argmax(scores, dim=-1)
        scores_net = eye(scores.size(-1), device=scores.device)
        return scores_net[idx]

    @staticmethod
    def backward(ctx, g):
        # pass the upstream gradient as is
        return g


class ProbabilisticSelection(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        """Sample the weight for an edge according to a multinomial distribution of the scores"""
        size = scores.size()
        prob = F.softmax(scores, dim=-1)
        idx = multinomial(prob.view(-1, size[-1]), num_samples=1, replacement=False).view(size[:-1])
        scores_net = eye(scores.size(-1), device=scores.device)
        return scores_net[idx]

    @staticmethod
    def backward(ctx, g):
        # pass the upstream gradient as is
        return g
