from torchvision.datasets import CIFAR100 as C100
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((.4914, .4822, .4465), (.2023, .1994, .2010)),
    ])


class CIFAR100(BaseDataset):
    def ds_train(self):
        return C100(root='./data', train=True, download=True,
                    transform=MultiSample(aug_transform(32, base_transform)))

    def ds_clf(self):
        return C100(root='./data', train=True, download=True,
                    transform=base_transform())

    def ds_test(self):
        return C100(root='./data', train=False, download=True,
                    transform=base_transform())
