from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])


class ImageNet(BaseDataset):
    def ds_train(self):
        return ImageFolder(root='data/imagenet/train', transform=MultiSample(
            aug_transform(224, base_transform)))

    def ds_clf(self):
        return ImageFolder(root='data/imagenet224/train',
                           transform=base_transform())

    def ds_test(self):
        return ImageFolder(root='data/imagenet224/val',
                           transform=base_transform())
