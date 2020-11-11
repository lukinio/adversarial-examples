import numpy as np
import torch
from nupic.torch.modules import rezero_weights, update_boost_strength
from copy import deepcopy


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

def train_model(model, tr_loader, val_loader, optimizer, loss_fn, epochs=30,
                patience=3, sparse=False, device="cpu"):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None

    for epoch in range(1, epochs+1):
        learn_epoch(model, tr_loader, optimizer, loss_fn, sparse=sparse, device=device)
        if sparse: model.apply(update_boost_strength)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device=device)
        print(f"Epoch: {epoch} Validation Loss: {val_loss:.4f} accuracy = {val_acc:.4f}")

        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break


def adv_epoch(model, loader, optimizer, loss_fn, attack, sparse=False, device='cpu'):
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        noise = attack(model, inputs, targets, loss_fn)
        outputs = model(inputs+noise)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if sparse:
            model.apply(rezero_weights)


def adv_train(model, tr_loader, val_loader, optimizer, loss_fn, attack,
              epochs=30, patience=3, sparse=False, device="cpu"):

    epochs_no_improve, min_loss, best_model = 0, np.inf, None

    for epoch in range(1, epochs+1):
        adv_epoch(model, tr_loader, optimizer, loss_fn, attack, sparse=sparse, device=device)
        if sparse: model.apply(update_boost_strength)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device=device)
        print(f"Epoch: {epoch} Validation Loss: {val_loss:.4f} accuracy = {val_acc:.4f}")

        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            epochs_no_improve = 0
            best_model = deepcopy(model)
        else:
            epochs_no_improve += 1
            print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print('Early stopping!')
                model = best_model
                break
