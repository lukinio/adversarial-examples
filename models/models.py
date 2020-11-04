import torch
import torch.nn as nn
import torch.nn.functional as F
from nupic.torch.modules import SparseWeights, Flatten, SparseWeights2d, KWinners, KWinners2d
import numpy as np
from models.activation import *


class FCNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=10)
         )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class SparseFCNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.clf = nn.Sequential(
            SparseWeights(nn.Linear(784, 256), 0.1),
            KWinners(n=256, percent_on=0.5, boost_strength=1.4),
            SparseWeights(nn.Linear(256, 128), 0.1),
            KWinners(n=128, percent_on=0.1, boost_strength=1.4),
            SparseWeights(nn.Linear(128, 64), 0.1),
            KWinners(n=64, percent_on=0.5, boost_strength=1.4),
            SparseWeights(nn.Linear(64, 32), 0.1),
            KWinners(n=32, percent_on=0.5, boost_strength=1.4),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.clf(x)
        return x
    
    
class SparseCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            SparseWeights2d(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), 0.1),
            nn.BatchNorm2d(32),
            KWinners2d(channels=32, percent_on=0.5, boost_strength=1.4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SparseWeights2d(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 0.1),
            nn.BatchNorm2d(64),
            KWinners2d(channels=64, percent_on=0.5, boost_strength=1.4),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            SparseWeights(nn.Linear(in_features=64*6*6, out_features=600), 0.1),
            KWinners(n=600, percent_on=0.5, boost_strength=1.4),
            SparseWeights(nn.Linear(in_features=600, out_features=120), 0.1),
            KWinners(n=120, percent_on=0.5, boost_strength=1.4),
            nn.Linear(in_features=120, out_features=10)
         )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNNSparseFC(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            KWinners2d(channels=32, percent_on=0.1, boost_strength=1.4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            KWinners2d(channels=64, percent_on=0.1, boost_strength=1.4),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            SparseWeights(nn.Linear(in_features=64*6*6, out_features=600), 0.1),
            KWinners(n=600, percent_on=0.1, boost_strength=1.4),
            SparseWeights(nn.Linear(in_features=600, out_features=120), 0.1),
            KWinners(n=120, percent_on=0.1, boost_strength=1.4),
            nn.Linear(in_features=120, out_features=10)
         )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class DenseFlatReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*6*6, out_features=600),
            Lambda(FlattenReLU.apply),
            nn.Linear(in_features=600, out_features=120),
            Lambda(FlattenReLU.apply),
            nn.Linear(in_features=120, out_features=10)
         )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

