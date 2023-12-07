import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from dataset import MyDataset
import os


class Autoencoder(nn.Module):
    def __init__(self, size):
        super(Autoencoder, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

