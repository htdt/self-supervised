import torchvision.transforms as T


def aug_transform(crop, base_transform):
    return T.Compose(
        [
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomResizedCrop(crop, interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            base_transform(),
        ]
    )


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
