# https://github.com/rdevon/DIM/blob/master/cortex_DIM/configs/convnets.py
import torch
import torch.nn as nn


class DIM32(nn.Module):
    def __init__(self):
        super(DIM32, self).__init__()

        def conv(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
            )

        self.feat = nn.Sequential(
            conv(3, 64),
            conv(64, 128),
            conv(128, 256),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.feat(x)


class DIM64(nn.Module):
    def __init__(self):
        super(DIM64, self).__init__()

        def conv(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
            )

        self.feat = nn.Sequential(
            conv(3, 96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(96, 192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(192, 384),
            conv(384, 384),
            conv(384, 192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(192 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.feat(x)
