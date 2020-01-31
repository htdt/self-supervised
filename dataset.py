import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def base_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])


def aug_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(20),
        transforms.RandomGrayscale(p=0.25),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        base_transform(),
    ])


def test_transform():
    return transforms.Compose([
        transforms.FiveCrop(20),
        transforms.Lambda(
            lambda crops: torch.stack([
                base_transform()(crop) for crop in crops]))
    ])


class CopySample:
    def __init__(self, transform, c=2):
        self.num_clones = c
        self.transform = transform

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num_clones))


def get_loaders(bs):
    ts_train = CIFAR10(root='./data', train=True, download=True,
                       transform=CopySample(aug_transform()))
    loader_train = torch.utils.data.DataLoader(
        ts_train, batch_size=bs, shuffle=True, num_workers=5,
        pin_memory=True, drop_last=True)

    ts_clf = CIFAR10(root='./data', train=True, download=True,
                     transform=aug_transform())
    loader_clf = torch.utils.data.DataLoader(
        ts_clf, batch_size=1024, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=False)

    ts_test = CIFAR10(root='./data', train=False, download=True,
                      transform=test_transform())
    loader_test = torch.utils.data.DataLoader(
        ts_test, batch_size=1000, shuffle=False, num_workers=5)

    return loader_train, loader_clf, loader_test
