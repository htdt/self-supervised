from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )


class ImageNet(BaseDataset):
    def ds_train(self):
        t = MultiSample(aug_transform(224, base_transform))
        return ImageFolder(root="/imagenet/train", transform=t)

    def ds_clf(self):
        t = base_transform()
        return ImageFolder(root="/imagenet224/train", transform=t)

    def ds_test(self):
        t = base_transform()
        return ImageFolder(root="/imagenet224/val", transform=t)
