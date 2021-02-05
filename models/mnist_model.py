import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activation import DTLinear


class MnistBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistDTLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            DTLinear(784, 256),
            DTLinear(256, 128),
            DTLinear(128, 64),
            DTLinear(64, 10),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistBaseConvModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*8*8, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistDTLConvModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            DTLinear(16*8*8, 120),
            DTLinear(120, 60),
            DTLinear(60, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
