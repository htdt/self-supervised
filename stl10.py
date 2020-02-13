import torch
from torchvision.datasets import STL10
import torchvision.transforms as T
from transforms import WeightedRandomChoice, MultiSample


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.43, .42, .39), (.27, .26, .27))])


def crop64():
    return T.Compose([T.Resize(70, interpolation=3), T.CenterCrop(64)])


def test_transform():
    return T.Compose([crop64(), base_transform()])


def aug_transform(gp, jp, rp):
    return T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.RandomGrayscale(p=gp),
        T.RandomApply([T.ColorJitter(.4, .4, .4, .2)], p=jp),
        WeightedRandomChoice([
            T.RandomResizedCrop(
                64, scale=(.3, 1), ratio=(.7, 1.4), interpolation=3),
            crop64()], [rp, 1 - rp]),
        base_transform()
    ])


def loader_train(batch_size):
    t = [aug_transform(rp=.5, gp=.25, jp=.5),
         aug_transform(rp=.9, gp=.25, jp=.5)]
    ts_train = STL10(root='./data', split='train+unlabeled', download=True,
                     transform=MultiSample(t))
    return torch.utils.data.DataLoader(
        ts_train, batch_size=batch_size, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)


def loader_clf(aug=False, batch_size=1000):
    t = aug_transform(rp=.05, gp=.1, jp=.1) if aug else test_transform()
    ts_clf = STL10(root='./data', split='train', download=True, transform=t)
    return torch.utils.data.DataLoader(
        ts_clf, batch_size=batch_size, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = STL10(root='./data', split='test', download=True,
                    transform=test_transform())
    return torch.utils.data.DataLoader(
        ts_test, batch_size=batch_size, shuffle=False, num_workers=16)
