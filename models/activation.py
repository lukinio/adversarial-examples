import torch
import torch.nn as nn

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


class FlattenReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        ini = x.argsort(dim=1, descending=True)[:, :int(0.1*x.size(1))]
        thresholds = torch.Tensor([x[i, e].abs().sum() for i, e in enumerate(ini)]).to(device)
        self.save_for_backward(x, thresholds)
        for i, e in enumerate(x):
            e.clamp_(0, thresholds[i])
        return x

    @staticmethod
    def backward(self, grad_output):
        x, thresholds = self.saved_tensors
        grad = grad_output.clone()
        for i, (g, ini) in enumerate(zip(grad, x)):
            g[ini<0] = 0
            g[ini>thresholds[i]] = 0
        return grad_output
