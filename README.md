## Group Normalization in MXNet
[Group Normalization](https://arxiv.org/abs/1803.08494) (GN) was proposed by He Kaiming's team in March 2018. GN optimizes the disadvantage of BN in smaller mini-batch situations. 

Group Normalization (GN) is an alternative to BN. It first divides channels into groups, and then calculates the mean and method in each group for normalization. The calculation of GN is independent of Batch Size, and the accuracy is stable for different Batch Size. In addition, GN is easy to fine-tuning from the pre-trained model. The comparison between GN and BN is shown in the figure.


##### Group Normalization in TF [gn_tf](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_tf.py) 
##### Group Normalization in Pytorch
[gn_pytorch](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_pytorch.py) group normalization in Pytorch.
##### Group Normalization in MXNet Symbol
[gn_symbol](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_symbol.py) group normalization in MXNet Symbol.
##### Group Normalization in MXNet Gluon
[gn_gluon](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/gn_gluon.py) group normalization in MXNet Gluon.
##### Group Normalization in MXNet CustomOperator
[groupnormOp](https://github.com/jianzhnie/GroupNorm-MXNet/blob/master/groupnormOp.py) group normalization in MXNet Custom Operator.
