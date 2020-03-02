from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.480, .448, .398), (.277, .269, .282))])


class TinyImageNet(BaseDataset):
    def ds_train(self):
        t = MultiSample(aug_transform(64, base_transform))
        return ImageFolder(root='data/tiny-imagenet-200/train', transform=t)

    def ds_clf(self):
        return ImageFolder(root='data/tiny-imagenet-200/train',
                           transform=base_transform())

    def ds_test(self):
        return ImageFolder(root='data/tiny-imagenet-200/test',
                           transform=base_transform())
