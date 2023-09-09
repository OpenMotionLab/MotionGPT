import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, direct_weighting=False, no_std=False):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        # (bs, nfeats, nframe) <= (nframe, bs, nfeats)
        x = x.permute(1,2,0) 

        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        # self.weight = torch.ones_like(self.weight)

        if direct_weighting:
            x_reshaped = x.contiguous().view(b * c)
            if no_std:
                out = x_reshaped + self.bias
            else:
                out = x_reshaped.mul(self.weight) + self.bias
            out = out.view(b, c, *x.size()[2:])
        else:
            x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])        
            out = F.batch_norm(
                x_reshaped, running_mean, running_var, self.weight, self.bias,
                True, self.momentum, self.eps)
            out = out.view(b, c, *x.size()[2:])

        # (nframe, bs, nfeats) <= (bs, nfeats, nframe)
        out = out.permute(2,0,1) 
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[: , : m.num_features]
            std = adain_params[: , m.num_features: 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[: , 2 * m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params
