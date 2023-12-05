import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from dataset import MyDataset
import os


class Autoencoder(nn.Module):
    def __init__(self, size, sigmoid: bool = False):
        super(Autoencoder, self).__init__()

        self.sigmoid = sigmoid
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder1 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, size),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.sigmoid:
            x = self.decoder2(x)
        else:
            x = self.decoder1(x)
        return x

