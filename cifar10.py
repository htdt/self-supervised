import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from transforms import MultiSample, Resize2


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


def loader_train(batch_size):
    t = [aug_transform(s1=1.15, s2=1.4, rp=.5, gp=.25, jp=.5),
         aug_transform(s1=1.4, s2=2.1, rp=.9, gp=.25, jp=.5)]
    ts_train = CIFAR10(root='./data', train=True, download=True,
                       transform=MultiSample(t))
    return torch.utils.data.DataLoader(
        ts_train, batch_size=batch_size, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)


def loader_clf(aug=False, batch_size=1000):
    if not aug:
        t = base_transform()
    else:
        t = aug_transform(s1=1.15, s2=2.1, rp=.05, gp=.1, jp=.1)
    ts_clf = CIFAR10(root='./data', train=True, download=True,
                     transform=t)
    return torch.utils.data.DataLoader(
        ts_clf, batch_size=batch_size, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = CIFAR10(root='./data', train=False, download=True,
                      transform=base_transform())
    return torch.utils.data.DataLoader(
        ts_test, batch_size=batch_size, shuffle=False, num_workers=16)
