import numpy as np
import torch
from nupic.torch.modules import rezero_weights, update_boost_strength
from copy import deepcopy
from datetime import datetime as dt

checkpoint_path = "../saved/checkpoint.pt"

def learn_epoch(model, loader, optimizer, loss_fn, sparse=False, device="cpu"):
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if sparse:
            model.apply(rezero_weights)

def eval_epoch(model, loader, loss_fn, device="cpu"):
    total_loss, accuracy = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.data.item() * inputs.size(0)
            accuracy += (outputs.max(dim=1)[1] == targets).sum().item()

        total_loss /= len(loader.dataset)
        accuracy /= len(loader.dataset)

    return total_loss, accuracy

def train_model(model, tr_loader, val_loader, optimizer, loss_fn, scheduler=None,
                epochs=30, patience=4, sparse=False, device="cpu"):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None

    for epoch in range(1, epochs+1):
        start = dt.now().replace(microsecond=0)
        learn_epoch(model, tr_loader, optimizer, loss_fn, sparse=sparse, device=device)
        if sparse: model.apply(update_boost_strength)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device=device)
        if scheduler is not None:
            scheduler.step(val_loss)
        out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
        print(out.format(epoch, val_loss, val_acc, dt.now().replace(microsecond=0)-start))

        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break


def adv_epoch(model, loader, optimizer, loss_fn, attack, params, sparse=False, device='cpu'):
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        noise = attack(model, inputs, targets, loss_fn,
                       epsilon=params["epsilon"],
                       alpha=params["alpha"],
                       num_iter=params["num_iter"]
                       )
        outputs = model(inputs+noise)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if sparse:
            model.apply(rezero_weights)


def adv_train(model, tr_loader, val_loader, attack, attack_params, optimizer, loss_fn,
              scheduler=None, epochs=30, patience=4, sparse=False, device="cpu"):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None

    for epoch in range(1, epochs+1):
        start = dt.now().replace(microsecond=0)
        adv_epoch(model, tr_loader, optimizer, loss_fn, attack,
                  attack_params, sparse=sparse, device=device)
        if sparse: model.apply(update_boost_strength)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device=device)
        if scheduler is not None:
            scheduler.step(val_loss)
        out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
        print(out.format(epoch, val_loss, val_acc, dt.now().replace(microsecond=0)-start))

        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break
