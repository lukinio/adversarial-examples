import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Lambda(nn.Module):
    """
    Input: 
        A Function
    Returns : 
        A Module that can be used inside nn.Sequential
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    
    def extra_repr(self):
        return self.func.__qualname__.split(".")[0] + "()"

    
# class OLDFlattenReLU(torch.autograd.Function):
#     @staticmethod
#     def forward(self, x):
#         ini = x.argsort(dim=1, descending=True)[:, :int(0.1*x.size(1))]
#         thresholds = torch.Tensor([x[i, e].abs().sum() for i, e in enumerate(ini)]).to(device)
#         self.save_for_backward(x, thresholds)
#         for i, e in enumerate(x):
#             e.clamp_(0, thresholds[i])
#         return x

#     @staticmethod
#     def backward(self, grad_output):
#         x, thresholds = self.saved_tensors
#         grad = grad_output.clone()
#         for i, (g, ini) in enumerate(zip(grad, x)):
#             g[ini<0] = 0
#             g[ini>thresholds[i]] = 0
#         return grad_output

class FlattenReLU(torch.autograd.Function):
    
    @staticmethod
    def forward(self, x, thresholds=None):
        self.save_for_backward(x, thresholds)
        x = torch.where(x > thresholds, thresholds, x) 
        x[x<0] = 0
        return x

    @staticmethod
    def backward(self, grad_output):
        x, thresholds = self.saved_tensors
        x_grad = grad_output.clone()
        x_grad[x>thresholds] = 0
        x_grad[x<0] = 0
        return x_grad, None


class DTLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        xe = torch.stack([x]*self.weight.size(0), dim=1)
        we = torch.stack([self.weight]*x.size(0), dim=0)
        z, _ = (we * xe).topk(int(0.1*self.weight.size(1)))
        thresholds = z.sum(dim=2)
        x = F.linear(x, self.weight, self.bias)
        return FlattenReLU.apply(x, thresholds)
