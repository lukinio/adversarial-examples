import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        xe = torch.stack([x]*self.weight.size(0), dim=1)
        we = torch.stack([self.weight]*x.size(0), dim=0)
        z, _ = (we * xe).topk(int(0.2*self.weight.size(1)))
        thresholds = z.sum(dim=2)
        x = F.linear(x, self.weight, self.bias)
        return FlattenReLU.apply(x, thresholds)
