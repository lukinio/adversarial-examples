import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_images(X, y, yp, M, N):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    X, y, yp = X.cpu(), y.cpu(), yp.cpu()
    for i in range(M):
        for j in range(N):
#             ax[i][j].imshow(1-X[i*N+j][0].detach().numpy())
            ax[i][j].imshow(X[i*N+j].detach().permute(1,2,0))

            pred_label = yp[i*N+j].max(dim=0)[1]
            title = ax[i][j].set_title(f"{y[i*N+j]} -> {pred_label}")
            plt.setp(title, color=('g' if pred_label == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()


def plot_learn_curves(hists, labels, ylabel):
    x = np.arange(1, len(hists[0][0])+1)
    fig, axes = plt.subplots(nrows=1, ncols=len(hists))

    for i, (hist, yl) in enumerate(zip(hists, ylabel)):
        for h, l in zip(hist, labels):
            axes[i].plot(x, h, label=l)
            axes[i].legend(loc='best')
            axes[i].set_xlabel('epoch')
            axes[i].set_ylabel(yl)
            axes[i].set_title('model loss' if 'loss' in yl else 'model accuracy')

    fig.tight_layout(pad=3.0)
    plt.show()