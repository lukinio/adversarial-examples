import torch

class Lambda(torch.nn.Module):
    """
    Input: A Function
    Returns : A Module that can be used 
              inside nn.Sequential
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


class FlattenReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        x = torch.sort(input_.detach().clone(), descending=True)
        t = torch.sum(x[0][:int(0.1*input_.size(0))])

        self.save_for_backward(input_, t)
        return input_.clamp(min=0, max=t)

    @staticmethod
    def backward(self, grad_output):
        _, t = self.saved_tensors
        return grad_output.clamp(min=0, max=t)