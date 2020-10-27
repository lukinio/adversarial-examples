import numpy as np
import torch
from sklearn.metrics import accuracy_score
from nupic.torch.modules import rezero_weights, update_boost_strength

    
def train_epoch(model, optimizer, loss_fn, train_loader, sparse_net=False, device="cpu"):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        if sparse_net:
            model.apply(rezero_weights)
            
def test_epoch(model, val_loader, loss_fn, device="cpu"):
    val_loss = 0.0
    pred_labels, true_labels = [], []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets) 
            val_loss += loss.data.item() * inputs.size(0)

            true_labels = targets.numpy()
            _, pred_labels = torch.max(outputs, 1)

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(true_labels, pred_labels)
        
    return val_loss, val_acc


def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs=30,
                n_epochs_stop=3, sparse_net=False, device="cpu"):
    
    epochs_no_improve, min_loss = 0, np.inf
    for epoch in range(1, epochs+1):
        train_epoch(model, optimizer, loss_fn, train_loader, sparse_net=sparse_net, device=device)
        if sparse_net:
            model.apply(update_boost_strength)
        val_loss, val_acc = test_epoch(model, val_loader, loss_fn, device=device)
        print("Epoch: {} Validation Loss: {:.2f} accuracy = {:.2f}".format(epoch, val_loss, val_acc))
        
        if val_loss < min_loss:
            min_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "models/checkpoint.pt")
        else:
            epochs_no_improve += 1
            print("epochs_no_improve: {}/{}".format(epochs_no_improve, n_epochs_stop, min_loss))
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                model.load_state_dict(torch.load("models/checkpoint.pt", map_location=device))
                break
        