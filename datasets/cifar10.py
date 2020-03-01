from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from transforms import MultiSample, aug_transform


def base_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((.4914, .4822, .4465), (.2023, .1994, .2010)),
    ])


def loader_train(batch_size):
    t = MultiSample(aug_transform(32, base_transform))
    ts_train = CIFAR10(root='./data', train=True, download=True, transform=t)
    return DataLoader(ts_train, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_clf(batch_size=1000):
    ts_clf = CIFAR10(root='./data', train=True, download=True,
                     transform=base_transform())
    return DataLoader(ts_clf, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = CIFAR10(root='./data', train=False, download=True,
                      transform=base_transform())
    return DataLoader(ts_test, batch_size=batch_size, shuffle=False,
                      num_workers=8)
