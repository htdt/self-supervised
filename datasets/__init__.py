from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .stl10 import STL10
from .tiny_in import TinyImageNet
from .imagenet import ImageNet


def get_ds(name):
    if name == 'cifar10':
        return CIFAR10
    elif name == 'cifar100':
        return CIFAR100
    elif name == 'stl10':
        return STL10
    elif name == 'tiny_in':
        return TinyImageNet
    elif name == 'imagenet':
        return ImageNet
    else:
        raise Exception('invalid dataset name')
