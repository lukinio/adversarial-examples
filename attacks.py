import torch

def clip(x, x_, eps):
    lower_clip = torch.max(torch.stack([torch.zeros_like(x), x - eps, x_]), dim=0)[0]
    return torch.min(torch.stack([torch.ones_like(x), x + eps, lower_clip]), dim=0)[0]


def generate_noise(model, loss_fn, examples, targets, max_iter=1, alpha=1.0,
                   clip_eps=1/255, do_clip=False, minimize=False, iterative=True):

    direction = -1 if minimize else 1
    if iterative:
        alpha /= max_iter

    adv_ex = examples.clone()
    noise = torch.zeros_like(adv_ex, requires_grad=True)

    for _ in range(max_iter):
        noise.requires_grad = True
        model.zero_grad()

        loss = loss_fn(model(adv_ex+noise), targets)
        loss.backward()
        noise = direction * alpha * noise.grad.sign()

        if do_clip:
            adv_ex = clip(adv_ex, adv_ex+noise, clip_eps)
        else:
            adv_ex += noise

    return noise


def fgsm(model, loss_fn, examples, targets, alpha=0.1, max_iter=1):
    return generate_noise(model, loss_fn, examples, targets, max_iter=max_iter, alpha=alpha, iterative=False)


def bim(model, loss_fn, examples, targets, max_iter=5, alpha=0.1, clip_eps=1/255, do_clip=True):
    return generate_noise(model, loss_fn, examples, targets, max_iter=max_iter, alpha=alpha,
                          clip_eps=clip_eps, do_clip=do_clip, iterative=True)


def llc(model, loss_fn, examples, max_iter=10, alpha=0.1, clip_eps=8/255, do_clip=True):
    last_lilely_targets = model(examples).argmin(dim=1).detach()
    return generate_noise(model, loss_fn, examples, last_lilely_targets, max_iter=max_iter,
                          alpha=alpha, do_clip=True, clip_eps=clip_eps, minimize=True)
