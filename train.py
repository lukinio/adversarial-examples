
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.train_utils import train_model, adv_train
from models.resnet import ResNet, SparseResNet
from utils.attacks import pgd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tr_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
    ])
vl_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True)
    ])

ds = CIFAR10('../data', train=True, download=True, transform=tr_transform)
ds_test = CIFAR10('../data', train=False, download=True, transform=vl_transform)

batch_size = 400
train_dl = DataLoader(ds, batch_size, shuffle=True)
valid_dl = DataLoader(ds_test, batch_size, shuffle=True)


model = SparseResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Terning Adversarialny dla SparseResNet")
adv_train(model, train_dl, valid_dl, optimizer, loss_fn, pgd,
          epochs=30, sparse=True, device=device)
torch.save(model.state_dict(), "saved/sparse_resnet_robust.pt")


model = ResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Terning Adversarialny dla ResNet")
adv_train(model, train_dl, valid_dl, optimizer, loss_fn, pgd,
          epochs=30, device=device)
torch.save(model.state_dict(), "saved/resnet_robust.pt")


model = SparseResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Ucze model SparseResNet")
train_model(model, train_dl, valid_dl, optimizer, loss_fn,
            epochs=30, device=device)
torch.save(model.state_dict(), "saved/sparse_resnet.pt")


model = ResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Ucze model ResNet")
train_model(model, train_dl, valid_dl, optimizer, loss_fn,
            epochs=30, device=device)
torch.save(model.state_dict(), "saved/resnet.pt")


