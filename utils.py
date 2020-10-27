import torch
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt; plt.style.use('default')
import numpy as np
        
def plot_images(X, y, yp, M, N):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].detach().numpy(), cmap="gray")
            pred_label = yp[i*N+j].max(dim=0)[1]
            title = ax[i][j].set_title("{} -> {}".format(y[i*N+j], pred_label))
            plt.setp(title, color=('g' if pred_label == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    
    
