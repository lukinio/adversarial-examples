import re
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nupic.torch.modules import rezero_weights, update_boost_strength
import matplotlib.pyplot as plt


class ModelTrainer:

    checkpoint_path = "../saved/{}_checkpoint.pt"

    def __init__(self, train_dataset, test_dataset, is_sparse, batch_size=100):
        self.batch_size = batch_size
        self.train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_sparse = is_sparse
        self.logs = {}


    def _train_epoch(self, model, loss_fn, optimizer):
        loader_len = len(self.train_dl.dataset)
        correct, total_loss = 0., 0.

        for k, (X, y) in enumerate(self.train_dl, 1):
            start = dt.now().replace(microsecond=0)
            X, y = X.to(self.device), y.to(self.device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.is_sparse:
                model.apply(rezero_weights)

            correct += (outputs.argmax(dim=1) == y).sum().item()
            total_loss += loss.item() * X.size(0)
            end = dt.now().replace(microsecond=0)
            print(f"train iteration: {k}/{len(self.train_dl)} time: {end-start}", end="\r")
        print(" " * 80, end="\r")

        return total_loss / loader_len, correct / loader_len


    def _test_epoch(self, model, loss_fn):
        loader_len = len(self.test_dl.dataset)
        correct, total_loss = 0., 0.

        model.eval()
        with torch.no_grad():
            for k, (X, y) in enumerate(self.test_dl, 1):
                start = dt.now().replace(microsecond=0)
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                correct += (outputs.argmax(dim=1) == y).sum().item()
                total_loss += loss.item() * X.size(0)
                end = dt.now().replace(microsecond=0)
                print(f"test iteration: {k}/{len(self.test_dl)} time: {end-start}", end="\r")
            print(" " * 80, end="\r")

        return total_loss / loader_len, correct / loader_len


    def train(self, model, loss_fn, optimizer, scheduler=None, epochs=30, patience=4):
        self.logs = {'loss': {"train": [], "test": []}, 'accuracy': {"train": [], "test": []}}
        model = model.to(self.device)
        epochs_no_improve, min_loss = 0, np.inf
        model_name = re.sub(r'\W+', '', str(model.__class__).split(".")[-1])

        for e in range(1, epochs+1):
            start = dt.now().replace(microsecond=0)
            train_loss, train_acc = self._train_epoch(model, loss_fn, optimizer)
            if self.is_sparse:
                model.apply(update_boost_strength)
            test_loss, test_acc = self._test_epoch(model, loss_fn)
            if scheduler is not None:
                scheduler.step(test_loss)
            out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
            print(out.format(e, test_loss, test_acc, dt.now().replace(microsecond=0)-start))

            # Update logs
            self.logs['loss']["train"].append(train_loss)
            self.logs['loss']["test"].append(test_loss)
            self.logs['accuracy']["train"].append(train_acc)
            self.logs['accuracy']["test"].append(test_acc)

            # Early stopping
            if test_loss < min_loss:
                min_loss = test_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), ModelTrainer.checkpoint_path.format(model_name))
            else:
                epochs_no_improve += 1
                print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
                if epochs_no_improve == patience:
                    model.load_state_dict(torch.load(ModelTrainer.checkpoint_path.format(model_name),
                                                     map_location=self.device))
                    print('Early stopping!')
                    break

        return self.logs

    def _adv_epoch(self, model, attack, loss_fn, optimizer, **kwargs):
        loader_len = len(self.train_dl.dataset)
        correct, total_loss = 0., 0.

        for k, (X, y) in enumerate(self.train_dl, 1):
            start = dt.now().replace(microsecond=0)
            X, y = X.to(self.device), y.to(self.device)
            delta = attack(model, X, y, loss_fn, **kwargs)
            outputs = model(X+delta)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.is_sparse:
                model.apply(rezero_weights)

            correct += (outputs.argmax(dim=1) == y).sum().item()
            total_loss += loss.item() * X.size(0)
            end = dt.now().replace(microsecond=0)
            print(f"adv train iteration: {k}/{len(self.train_dl)} time: {end-start}", end="\r")
        print(" " * 80, end="\r")

        return total_loss / loader_len, correct / loader_len

    def adv_train(self, model, attack, params, loss_fn, optimizer, scheduler=None, epochs=30, patience=4):
        self.logs = {'loss': {"adv_train": [], "test": []}, 'accuracy': {"adv_train": [], "test": []}}
        model = model.to(self.device)
        epochs_no_improve, min_loss = 0, np.inf
        model_name = re.sub(r'\W+', '', str(model.__class__).split(".")[-1])

        for e in range(1, epochs+1):
            start = dt.now().replace(microsecond=0)
            train_loss, train_acc = self._adv_epoch(model, attack, loss_fn, optimizer, **params)
            if self.is_sparse:
                model.apply(update_boost_strength)
            test_loss, test_acc = self._test_epoch(model, loss_fn)
            if scheduler is not None:
                scheduler.step(test_loss)
            out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, time: {}"
            print(out.format(e, test_loss, test_acc, dt.now().replace(microsecond=0)-start))

            # Update logs
            self.logs['loss']["adv_train"].append(train_loss)
            self.logs['loss']["test"].append(test_loss)
            self.logs['accuracy']["adv_train"].append(train_acc)
            self.logs['accuracy']["test"].append(test_acc)

            # Early stopping
            if test_loss < min_loss:
                min_loss = test_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), ModelTrainer.checkpoint_path.format(model_name))
            else:
                epochs_no_improve += 1
                print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
                if epochs_no_improve == patience:
                    model.load_state_dict(torch.load(ModelTrainer.checkpoint_path.format(model_name),
                                                     map_location=self.device))
                    print('Early stopping!')
                    break

        return self.logs


def plot_history(hists):
    x = np.arange(1, len(hists["loss"]["test"])+1)
    _, axes = plt.subplots(nrows=1, ncols=len(hists), figsize=(15, 5))
    for ax, (name, hist) in zip(axes, hists.items()):
        for label, h in hist.items():
            ax.plot(x, h, label=label)

        ax.set_title("Model " + name)
        ax.set_xlabel('epochs')
        ax.set_ylabel(name)
        ax.legend(loc="best")

    plt.show()
