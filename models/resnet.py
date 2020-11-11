import torch
import torch.nn as nn
import torch.nn.functional as F
from nupic.torch.modules import SparseWeights, KWinners


class BaseResNet(nn.Module):
    def __init__(self, classifier):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)
        )

        self.classifier = classifier

    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def ResNet():
    classifier = nn.Sequential(
        nn.Linear(512*4*4, 4000),
        nn.ReLU(),
        nn.Linear(4000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return BaseResNet(classifier)

def SparseResNet():
    classifier = nn.Sequential(
        SparseWeights(nn.Linear(512*4*4, 4000), 0.1),
        KWinners(n=4000, percent_on=0.1, boost_strength=1.4),

        SparseWeights(nn.Linear(4000, 2000), 0.1),
        KWinners(n=2000, percent_on=0.1, boost_strength=1.4),

        SparseWeights(nn.Linear(2000, 1000), 0.1),
        KWinners(n=1000, percent_on=0.1, boost_strength=1.4),

        SparseWeights(nn.Linear(1000, 100), 0.1),
        KWinners(n=100, percent_on=0.10, boost_strength=1.4),

        nn.Linear(100, 10)
    )
    return BaseResNet(classifier)
