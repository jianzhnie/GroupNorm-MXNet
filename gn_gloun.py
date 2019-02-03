class GroupNorm(nn.HybridBlock):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5,
                 multi_precision=False, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)

        with self.name_scope():
            self.weight = self.params.get('weight', grad_req='write',
                                          shape=(1, num_channels, 1, 1))
            self.bias = self.params.get('bias', grad_req='write',
                                        shape=(1, num_channels, 1, 1))
        self.C = num_channels
        self.G = num_groups
        self.eps = eps
        self.multi_precision = multi_precision

        assert self.C % self.G == 0

    def hybrid_forward(self, F, x, weight, bias):

        x_new = F.reshape(x, (0, self.G, -1))                                # (N,C,H,W) -> (N,G,H*W*C//G)

        if self.multi_precision:
            mean = F.mean(F.cast(x_new, "float32"),
                          axis=-1, keepdims=True)                            # (N,G,H*W*C//G) -> (N,G,1)
            mean = F.cast(mean, "float16")
        else:
            mean = F.mean(x_new, axis=-1, keepdims=True)

        centered_x_new = F.broadcast_minus(x_new, mean)                      # (N,G,H*W*C//G)

        if self.multi_precision:
            var = F.mean(F.cast(F.square(centered_x_new),"float32"),
                         axis=-1, keepdims=True)                             # (N,G,H*W*C//G) -> (N,G,1)
            var = F.cast(var, "float16")
        else:
            var = F.mean(F.square(centered_x_new), axis=-1, keepdims=True)

        x_new = F.broadcast_div(centered_x_new, F.sqrt(var + self.eps)       # (N,G,H*W*C//G) -> (N,C,H,W)
                                ).reshape_like(x)
        x_new = F.broadcast_add(F.broadcast_mul(x_new, weight),bias)
        return x_new
