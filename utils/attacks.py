import torch

def fgsm(model, X, y, loss_fn, epsilon=0.1):
    noise = torch.zeros_like(X, requires_grad=True)
    loss = loss_fn(model(X + noise), y)
    loss.backward()
    return epsilon * noise.grad.detach().sign()

def pgd(model, X, y, loss_fn, epsilon=0.1, alpha=0.01, num_iter=20):
    noise = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = loss_fn(model(X + noise), y)
        loss.backward()
        noise.data = (noise + alpha*noise.grad.detach().sign()).clamp(-epsilon, epsilon)
        noise.grad.zero_()
    return noise.detach()