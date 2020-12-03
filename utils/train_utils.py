import re
from copy import deepcopy
from datetime import datetime as dt
import numpy as np
import torch
from nupic.torch.modules import rezero_weights, update_boost_strength
from utils.utils import plot_learn_curves

checkpoint_path = "../saved/{}_checkpoint.pt"


def epoch(model, loader, loss_fn, optimizer=None, sparse=False, device="cpu"):
    """training/evaluation epoch over the dataset"""
    if optimizer is not None:
        phase_name, phase = "train", torch.enable_grad()
    else:
        phase_name, phase = "test", torch.no_grad()

    total_loss, accuracy = 0., 0.
    with phase:
        for k, (inputs, targets) in enumerate(loader, 1):
            start = dt.now().replace(microsecond=0)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if phase_name == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if sparse:
                    model.apply(rezero_weights)

            total_loss += loss.data.item() * inputs.size(0)
            accuracy += (outputs.max(dim=1)[1] == targets).sum().item()
            end = dt.now().replace(microsecond=0)
            print(f"{phase_name} iter: {k}/{len(loader)} time: {end-start}", end="\r")
        print(" " * 80, end="\r")

    return total_loss / len(loader.dataset), accuracy / len(loader.dataset)


def train(model, tr_loader, vl_loader, loss_fn, optimizer, scheduler=None,
          epochs=30, patience=4, sparse=False, device="cpu", curves=True):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None
    tr_loss_hist, tr_acc_hist = [], []
    vl_loss_hist, vl_acc_hist = [], []
    model_name = re.sub(r'\W+', '', str(model.__class__).split(".")[-1])

    for i in range(1, epochs+1):
        start = dt.now().replace(microsecond=0)
        tr_loss, tr_acc = epoch(model, tr_loader, loss_fn, optimizer, sparse=sparse, device=device)
        if sparse: model.apply(update_boost_strength)
        vl_loss, vl_acc = epoch(model, vl_loader, loss_fn, sparse=sparse, device=device)
        if scheduler is not None:
            scheduler.step(vl_loss)
        out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
        print(out.format(i, vl_loss, vl_acc, dt.now().replace(microsecond=0)-start))
        
#       learning curves
        tr_loss_hist.append(tr_loss)
        tr_acc_hist.append(tr_acc)
        vl_loss_hist.append(vl_loss)
        vl_acc_hist.append(vl_acc)

        # Early stopping
        if vl_loss < min_loss:
            min_loss = vl_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
            torch.save(model.state_dict(), checkpoint_path.format(model_name))
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break
    if curves:
        loss_hist = (tr_loss_hist, vl_loss_hist)
        acc_hist = (tr_acc_hist, vl_acc_hist)
        y_label = ("loss", "accuracy")
        label = ("train", "test")
        plot_learn_curves((loss_hist, acc_hist), label, y_label)

        
def adv_eval_epoch(model, attack, loader, loss_fn, sparse=False, device="cpu", **kwargs):
    """evaluation adversarial epoch over the dataset"""

    model.eval()
    total_loss, accuracy = 0., 0.
    for k, (inputs, targets) in enumerate(loader, 1):
        start = dt.now().replace(microsecond=0)
        inputs, targets = inputs.to(device), targets.to(device)
        noise = attack(model, inputs, targets, loss_fn, **kwargs)
        outputs = model(inputs+noise)
        loss = loss_fn(outputs, targets)
            
        total_loss += loss.data.item() * inputs.size(0)
        accuracy += (outputs.max(dim=1)[1] == targets).sum().item()
        end = dt.now().replace(microsecond=0)
        print(f"adv test iter: {k}/{len(loader)} time: {end-start}", end="\r")
    print(" " * 80, end="\r")

    return total_loss / len(loader.dataset), accuracy / len(loader.dataset)

def adv_epoch(model, attack, loader, loss_fn, optimizer, sparse=False, device="cpu", **kwargs):
    """training adversarial epoch over the dataset"""

    total_loss, accuracy = 0., 0.
    adv_loss, adv_accuracy = 0., 0.
    for k, (inputs, targets) in enumerate(loader, 1):
        start = dt.now().replace(microsecond=0)
        inputs, targets = inputs.to(device), targets.to(device)
        noise = attack(model, inputs, targets, loss_fn, **kwargs)
        outputs = model(inputs+noise)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sparse:
            model.apply(rezero_weights)
            
        total_loss += loss.data.item() * inputs.size(0)
        accuracy += (outputs.max(dim=1)[1] == targets).sum().item()
        end = dt.now().replace(microsecond=0)
        print(f"adv train iter: {k}/{len(loader)} time: {end-start}", end="\r")
    print(" " * 80, end="\r")
    
    return total_loss / len(loader.dataset), accuracy / len(loader.dataset)


def adv_train(model, attack, params, tr_loader, vl_loader, loss_fn, optimizer, scheduler=None,
              epochs=30, patience=4, sparse=False, device="cpu", curves=True):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None
    tr_loss_hist, tr_acc_hist = [], []
    adv_loss_hist, adv_acc_hist = [], []
    vl_loss_hist, vl_acc_hist = [], []
    model_name = re.sub(r'\W+', '', str(model.__class__).split(".")[-1])

    for i in range(1, epochs+1):
        start = dt.now().replace(microsecond=0)
        loss, acc = adv_epoch(model, attack, tr_loader, loss_fn, optimizer,
                              sparse=sparse, device=device, **params)
        if sparse: model.apply(update_boost_strength)
        vl_loss, vl_acc = epoch(model, vl_loader, loss_fn, optimizer=None, sparse=sparse, device=device)
        if scheduler is not None:
            scheduler.step(vl_loss)
        out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
        print(out.format(i, vl_loss, vl_acc, dt.now().replace(microsecond=0)-start))
        
#       learning curves
        tr_loss_hist.append(loss[0])
        adv_loss_hist.append(loss[1])
        vl_loss_hist.append(vl_loss)
        tr_acc_hist.append(acc[0])
        adv_acc_hist.append(acc[1])
        vl_acc_hist.append(vl_acc)

#         Early stopping
        if vl_loss < min_loss:
            min_loss = vl_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
            torch.save(model.state_dict(), checkpoint_path.format("robust_"+model_name))
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break
    
    if curves:
        loss_hist = (tr_loss_hist, adv_loss_hist, vl_loss_hist)
        acc_hist = (tr_acc_hist, adv_acc_hist, vl_acc_hist)
        y_label = ("loss", "accuracy")
        label = ("train", "adv_train", "test")
        plot_learn_curves((loss_hist, acc_hist), label, y_label)
