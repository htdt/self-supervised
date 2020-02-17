from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
from transforms import RandomApply2, MultiSample


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.43, .42, .39), (.27, .26, .27))])


def crop64():
    return T.Compose([T.Resize(70, interpolation=3), T.CenterCrop(64)])


def test_transform():
    return T.Compose([crop64(), base_transform()])


def aug_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.RandomGrayscale(p=.25),
        T.RandomApply([T.ColorJitter(.4, .4, .4, .2)], p=.5),
        RandomApply2(T.RandomResizedCrop(64, scale=(.25, 1), interpolation=3),
                     crop64(), p=.8),
        base_transform()
    ])


def loader_train(batch_size):
    t = MultiSample(aug_transform())
    ts_train = STL10(
        root='./data', split='train+unlabeled', download=True, transform=t)
    return DataLoader(ts_train, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_clf(batch_size=1000):
    ts_clf = STL10(root='./data', split='train', download=True,
                   transform=test_transform())
    return DataLoader(ts_clf, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = STL10(root='./data', split='test', download=True,
                    transform=test_transform())
    return DataLoader(ts_test, batch_size=batch_size, shuffle=False,
                      num_workers=8)
