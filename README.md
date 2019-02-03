## Group Normalization in MXNet
[Group Normalization](https://arxiv.org/abs/1803.08494) (GN) was proposed by He Kaiming's team in March 2018. GN optimizes the disadvantage of BN in smaller mini-batch situations. Group Normalization (GN) is an alternative to BN. It first divides channels into groups, and then calculates the mean and method in each group for normalization. The calculation of GN is independent of Batch Size, and the accuracy is stable for different Batch Size. In addition, GN is easy to fine-tuning from the pre-trained model. The comparison between GN and BN is shown in the figure.

![gn](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn.png)

##### Group Normalization in TF : [gn_tf.py](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_tf.py) 
##### Group Normalization in Pytorch : [gn_pytorch.py](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_pytorch.py) 
##### Group Normalization in MXNet Symbol : [gn_symbol.py](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_symbol.py)
##### Group Normalization in MXNet Gluon : [gn_gluon.py](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_gluon.py)
##### Group Normalization in MXNet CustomOperator : [groupnormOp.py](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/groupnormOp.py)

### How to use
Here is an example show how to replace the BN in your network with GN in the resnet bottleneck.

```
    def residual_unit(self, data, num_filter, gn_channel, stride, dim_match, name):
        bn1  = GroupNorm(data=data, in_channel = gn_channel, name=name + '_gn1')
        act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True, workspace=self.workspace, name=name + '_conv1')
        bn2   = GroupNorm(data=conv1, in_channel = int(num_filter*0.25), name=name+'_gn2')
        act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=self.workspace, name=name + '_conv2')
        bn3    = GroupNorm(data = conv2, in_channel=int(num_filter * 0.25), name=name+'_gn3')
        act3  = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=int(num_filter), kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                                   workspace=self.workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=int(num_filter), kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=self.workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
        return sum
```
