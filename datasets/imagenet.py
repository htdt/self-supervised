import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import ImageFilter
from .transforms import MultiSample, aug_transform
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


class ImageNet(BaseDataset):
    def ds_train(self):
        aug_with_blur = aug_transform(
            224,
            base_transform,
            self.aug_cfg,
            extra_t=[T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5)],
        )
        t = MultiSample(aug_with_blur, n=self.aug_cfg.num_samples)
        return ImageFolder(root=self.aug_cfg.imagenet_path + "train", transform=t)

    def ds_clf(self):
        t = base_transform()
        return ImageFolder(root=self.aug_cfg.imagenet_path + "clf", transform=t)

    def ds_test(self):
        t = base_transform()
        return ImageFolder(root=self.aug_cfg.imagenet_path + "test", transform=t)
