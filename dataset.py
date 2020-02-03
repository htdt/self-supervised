import random
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Resize2:
    def __init__(self, s1, s2, cs):
        self.s1, self.s2 = s1, s2
        self.crop_fn = T.RandomCrop(cs)

    def __call__(self, x):
        ds = max(0, self.s2 - self.s1)
        size = self.s1 + ds * random.random()
        size = round(size * x.size[0])
        x = TF.resize(x, size)
        x = self.crop_fn(x)
        return x


def base_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def aug_transform(s1, s2, rp, gp, jp):
    return T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.RandomGrayscale(p=gp),
        T.RandomApply([T.ColorJitter(.4, .4, .4, .2)], p=jp),
        T.RandomApply([Resize2(s1, s2, 32)], p=rp),
        base_transform()
    ])


def test_transform():
    return T.Compose([
        T.FiveCrop(20),
        T.Lambda(lambda cs: torch.stack([base_transform()(c) for c in cs]))
    ])


class MultiSample:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(t(x) for t in self.transforms)


def get_loaders(bs, aug0, aug1):
    t = [aug_transform(**aug0), aug_transform(**aug1)]
    ts_train = CIFAR10(root='./data', train=True, download=True,
                       transform=MultiSample(t))
    loader_train = torch.utils.data.DataLoader(
        ts_train, batch_size=bs, shuffle=True, num_workers=5,
        pin_memory=True, drop_last=True)

    ts_clf = CIFAR10(root='./data', train=True, download=True,
                     transform=aug_transform(**aug0))
    loader_clf = torch.utils.data.DataLoader(
        ts_clf, batch_size=1000, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)

    ts_test = CIFAR10(root='./data', train=False, download=True,
                      transform=base_transform())
    loader_test = torch.utils.data.DataLoader(
        ts_test, batch_size=1000, shuffle=False, num_workers=5)

    return loader_train, loader_clf, loader_test
