import torchvision.transforms as T


def aug_transform(crop, base_transform):
    return T.Compose([
        T.RandomApply([T.ColorJitter(.4, .4, .4, .1)], p=.8),
        T.RandomGrayscale(p=.2),
        T.RandomResizedCrop(crop, interpolation=3),
        T.RandomHorizontalFlip(p=.5),
        base_transform()
    ])


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
