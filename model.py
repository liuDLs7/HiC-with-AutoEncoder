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
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
            nn.Linear(256, size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

