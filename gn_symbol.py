def GroupNorm(self, data, in_channel, name, num_groups=32, eps=1e-5):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1] # G: number of groups for GN
    C = in_channel
    G = num_groups
    G = min(G, C)
    x_group= mx.sym.reshape(data = data, shape = (1, G, C//G, 0, -1))
    mean = mx.sym.mean(x_group, axis= (2, 3, 4), keepdims = True) 
    differ = mx.sym.broadcast_minus(lhs = x_group, rhs = mean)
    var = mx.sym.mean(mx.sym.square(differ), axis = (2, 3, 4), keepdims =True)
    x_groupnorm = mx.sym.broadcast_div(lhs = differ, rhs = mx.sym.sqrt(var + eps))
    #x_out = mx.sym.reshape(x_groupnorm, shape = (0, -3, -2)) 
    x_out = mx.sym.reshape_like(x_groupnorm, data)
    gamma = mx.sym.Variable(name = name + '_gamma',shape = (1,C,1,1), dtype='float32')
    beta = mx.sym.Variable(name = name + '_beta', shape=(1,C,1,1), dtype='float32')
    gn_x = mx.sym.broadcast_mul(lhs = x_out, rhs = gamma)
    gn_x = mx.sym.broadcast_plus(lhs = gn_x, rhs = beta)
    return gn_x
