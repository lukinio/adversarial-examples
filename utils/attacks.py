import torch

def clip(x, x_, eps):
    lower_clip = torch.max(torch.stack([torch.zeros_like(x), x - eps, x_]), dim=0)[0]
    return torch.min(torch.stack([torch.ones_like(x), x + eps, lower_clip]), dim=0)[0]

def fgsm(model, X, y, loss_fn, epsilon=0.1):
    noise = torch.zeros_like(X, requires_grad=True)
    loss = loss_fn(model(X + noise), y)
    loss.backward()
    return epsilon * noise.grad.detach().sign()

def pgd(model, X, y, loss_fn, epsilon=8/255, alpha=2/255, num_iter=10):
    noise = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        noise.requires_grad = True
        loss = loss_fn(model(X + noise), y)
        loss.backward()
        noise.data = (noise + alpha*noise.grad.detach().sign()).clamp(-epsilon, epsilon)
        noise.grad.zero_()
        adv_ex = clip(X, X+noise, epsilon)
        noise.data = adv_ex - X
    return noise.detach()
