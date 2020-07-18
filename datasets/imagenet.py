import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import ImageFilter
from .transforms import MultiSample
from .base import BaseDataset


class RandomBlur:
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )


def aug_transform():
    return T.Compose(
        [
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomResizedCrop(224, interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5),  # 1, 20
            base_transform(),
        ]
    )


class ImageNet(BaseDataset):
    def ds_train(self):
        t = MultiSample(aug_transform(), n=self.aug_cfg.num_samples)
        return ImageFolder(root="/imagenet/train", transform=t)

    def ds_clf(self):
        t = base_transform()
        return ImageFolder(root="/imagenet224/train", transform=t)

    def ds_test(self):
        t = base_transform()
        return ImageFolder(root="/imagenet224/val", transform=t)
