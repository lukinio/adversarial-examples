import torch
import torch.nn as nn
import torch.nn.functional as F
from nupic.torch.modules import SparseWeights, KWinners, KWinners2d
from models.activation import DTLinear


class FResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )

        self.classifier = nn.Sequential(
            DTLinear(512*4*4, 4000),
            DTLinear(4000, 2000),
            DTLinear(2000, 1000),
            DTLinear(1000, 100),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class FSparseResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            KWinners2d(64, percent_on=0.1, local=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            KWinners2d(128, percent_on=0.1, local=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            KWinners2d(128, percent_on=0.1, local=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            KWinners2d(128, percent_on=0.1, local=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            KWinners2d(256, percent_on=0.1, local=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            KWinners2d(512, percent_on=0.1, local=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            KWinners2d(512, percent_on=0.1, local=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            KWinners2d(512, percent_on=0.1, local=True),
            nn.MaxPool2d(4)
        )

        self.classifier = classifier = nn.Sequential(
            SparseWeights(DTLinear(512*4*4, 4000), 0.1),
            SparseWeights(DTLinear(4000, 2000), 0.1),
            SparseWeights(DTLinear(2000, 1000), 0.1),
            SparseWeights(DTLinear(1000, 100), 0.1),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
