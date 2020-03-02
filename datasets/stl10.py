from torchvision.datasets import STL10 as S10
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.43, .42, .39), (.27, .26, .27))])


def test_transform():
    return T.Compose([
        T.Resize(70, interpolation=3), T.CenterCrop(64), base_transform()])


class STL10(BaseDataset):
    def ds_train(self):
        return S10(root='./data', split='train+unlabeled', download=True,
                   transform=MultiSample(aug_transform(64, base_transform)))

    def ds_clf(self):
        return S10(root='./data', split='train', download=True,
                   transform=test_transform())

    def ds_test(self):
        return S10(root='./data', split='test', download=True,
                   transform=test_transform())
