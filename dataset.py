import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from transforms import Resize2


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


class MultiSample:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(t(x) for t in self.transforms)


def get_loader_train(bs, aug0, aug1):
    t = [aug_transform(**aug0), aug_transform(**aug1)]
    ts_train = CIFAR10(root='./data', train=True, download=True,
                       transform=MultiSample(t))
    return torch.utils.data.DataLoader(
        ts_train, batch_size=bs, shuffle=True, num_workers=5,
        pin_memory=True, drop_last=True)


def get_loader_clf(aug=None, bs=1000):
    t = base_transform() if aug is None else aug_transform(**aug)
    ts_clf = CIFAR10(root='./data', train=True, download=True,
                     transform=t)
    return torch.utils.data.DataLoader(
        ts_clf, batch_size=bs, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)


def get_loader_test(bs=1000):
    ts_test = CIFAR10(root='./data', train=False, download=True,
                      transform=base_transform())
    return torch.utils.data.DataLoader(
        ts_test, batch_size=bs, shuffle=False, num_workers=5)
