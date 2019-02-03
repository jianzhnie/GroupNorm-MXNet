import torch
import torch.nn as nn

from torch.nn import Parameter


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class GroupNormMoving(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5,
                 momentum=0.1, affine=True,
                 track_running_stats=True
                 ):
        super(GroupNormMoving, self).__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

        self.momentum = momentum
        self.affine = affine

        self.track_running_stats = track_running_stats

        tensor_shape = (1, num_features, 1, 1)

        if self.affine:
            self.weight = Parameter(torch.Tensor(*tensor_shape))
            self.bias = Parameter(torch.Tensor(*tensor_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            #     self.register_buffer('running_mean', torch.zeros(*tensor_shape))
            #     self.register_buffer('running_var', torch.ones(*tensor_shape))
            # else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, "Channel must be divided by groups"

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        if self.running_mean is None or self.running_mean.size() != mean.size():
            # self.running_mean = Parameter(torch.Tensor(mean.data.clone()))
            # self.running_var = Parameter(torch.Tensor(var.data.clone()))
            self.running_mean = Parameter(torch.Tensor(mean.data))
            self.running_var = Parameter(torch.Tensor(mean.data))

        if self.training and self.track_running_stats:
            self.running_mean.data = mean * self.momentum + \
                                     self.running_mean.data * (1 - self.momentum)
            self.running_var.data = var * self.momentum + \
                                    self.running_var.data * (1 - self.momentum)

        # mean = self.running_mean
        # var = self.running_var

        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None and self.running_var is not None:
                self.running_mean.zero_()
                self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))
