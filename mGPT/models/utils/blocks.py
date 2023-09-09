import torch
import torch.nn as nn
import torch.nn.functional as F
from mGPT.models.notused import AdaptiveInstanceNorm1d


class MLP(nn.Module):

    def __init__(self, cfg, out_dim, is_init):
        super(MLP, self).__init__()
        dims = cfg.MODEL.MOTION_DECODER.MLP_DIM
        n_blk = len(dims)
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim, norm='none', acti='none')
        self.model = nn.Sequential(*layers)

        if is_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.weight, 1)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def ZeroPad1d(sizes):
    return nn.ConstantPad1d(sizes, 0)


def get_acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        # return [nn.InstanceNorm1d(norm_dim, affine=False)]  # for rt42!
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvLayers(kernel_size,
               in_channels,
               out_channels,
               stride=1,
               pad_type='reflect',
               use_bias=True):
    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [
        pad((pad_l, pad_r)),
        nn.Conv1d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  bias=use_bias)
    ]


def ConvBlock(kernel_size,
              in_channels,
              out_channels,
              stride=1,
              pad_type='reflect',
              dropout=None,
              norm='none',
              acti='lrelu',
              acti_first=False,
              use_bias=True,
              inplace=True):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = ConvLayers(kernel_size,
                        in_channels,
                        out_channels,
                        stride=stride,
                        pad_type=pad_type,
                        use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers
