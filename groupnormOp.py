import numpy as np
import mxnet as mx

class GroupNorm(mx.operator.CustomOp):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    def __init__(self, gamma, beta, num_groups=32, eps=1e-5,**kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta
        self.G = num_groups
        self.eps = eps
        self.mean = None
        self.var = None
        self.x_norm = None


    def forward(self, is_train, req, in_data, out_data, aux):
        """
        Computes the forward pass for spatial group normalization.
        In contrast to layer normalization, group normalization splits each entry 
        in the data into G contiguous pieces, which it then normalizes independently.
        Per feature shifting and scaling are then applied to the data, in a manner identical 
        to that of batch normalization and layer normalization.


        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - G: Integer mumber of groups to split into, should be a divisor of C
        - gn_param: Dictionary with the following keys:
        - eps: Constant for numeric stability   
        
        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)                                        
        """

        x = in_data[0]
        N,C,H,W = x.shape
        # group the channel by G
        x_group = x.reshape((N, self.G, -1, H, W))
        self.mean = mx.nd.mean(x_group, axis=(2, 3, 4), keepdims=True) 
        self.var = mx.nd.mean((x_group self.mean)**2, axis = (2, 3, 4), keepdims=True) 
        # Normalization
        x_groupnorm = (x_group - self.mean) / mx.nd.sqrt(self.var + self.eps)
        # reshape to (N,C,H,W)
        self.x_norm = x_groupnorm.reshape((N,C,H,W))
        # output the group normalization result
        x_gn = self.x_norm * self.gamma + self.beta
        self.assign(out_data[0], req[0], x_gn)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """
        Implement the backward pass for spatial group normalization.      
        This will be extremely similar to the layer norm implementation. 

        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        
        """       
        dx, dgamma, dbeta = None, None, None 
        x = in_data[0]
        dout = out_grad[0]
        N,C,H,W = dout.shape
        # dbeta, dgamma 
        dbeta = mx.nd.sum(dout, axis=(0,2,3), keepdims=True) 
        dgamma = mx.nd.sum(dout*self.x_norm, axis=(0,2,3), keepdims=True)
        # get dx_group,(N, G, C // G, H, W)
        # dx_groupnorm
        dx_norm = dout * self.gamma 
        dx_groupnorm = dx_norm.reshape((N, self.G, C // self.G, H, W)) 
        # dvar
        x_group = x.reshape((N, self.G, C //self.G, H, W))
        dvar = mx.nd.sum(dx_groupnorm * -1.0 / 2 * (x_group - self.mean) / (self.var + self.eps) ** (3.0 / 2), axis=(2,3,4), keepdims=True)
        # dmean
        N_GROUP = C//self.G*H*W
        dmean1 = mx.nd.sum(dx_groupnorm * -1.0 / mx.nd.sqrt(self.var + self.eps), axis=(2,3,4), keepdims=True)
        dmean2_var = dvar * -2.0 / N_GROUP * mx.nd.sum(x_group - self.mean, axis=(2,3,4), keepdims=True)
        dmean = dmean1 + dmean2_var
        # dx_group
        dx_group1 = dx_groupnorm * 1.0 / mx.nd.sqrt(self.var + self.eps)
        dx_group2_mean = dmean * 1.0 / N_GROUP
        dx_group3_var = dvar * 2.0 / N_GROUP * (x_group - self.mean)
        dx_group = dx_group1 + dx_group2_mean + dx_group3_var
        # reshape 
        dx = dx_group.reshape((N, C, H, W)) 
        self.assign(in_grad[0], req[0], dx)

@mx.operator.register("groupnorm")  # register with name "groupnorm"
class GroupNormProp(mx.operator.CustomOpProp):
    def __init__(self, gamma, beta, num_groups=32, eps=1e-5,**kwargs):
        super(GroupNormProp, self).__init__(need_top_grad=True)
        """
        All arguments are in string format so you need
        to convert them back to the type you want.
        """
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.G = int(num_groups)
        self.eps = float(eps)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes
        output_shape = data_shape
        #return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return data_shape, output_shape, []

    def infer_type(self, in_type):
        dtype = in_type
        return (dtype), (dtype), ()

    def create_operator(self, ctx, shapes, dtypes):
        # create and return the CustomOp class.
        return GroupNorm(self.gamma, self.beta, self.G, self.eps)

    # def declare_backward_dependency(self,out_grad,in_data,out_data):
    #     return [out_grad[0]]
