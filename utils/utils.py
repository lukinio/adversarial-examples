import torch
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt; plt.style.use('default')
import numpy as np

def plot_images(X, y, yp, M, N):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    X, y, yp = X.cpu(), y.cpu(), yp.cpu()
    for i in range(M):
        for j in range(N):
#             ax[i][j].imshow(1-X[i*N+j][0].detach().numpy())
            ax[i][j].imshow(X[i*N+j].squeeze().permute(1, 2, 0))
            pred_label = yp[i*N+j].max(dim=0)[1]
            title = ax[i][j].set_title(f"{y[i*N+j]} -> {pred_label}")
            plt.setp(title, color=('g' if pred_label == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()


