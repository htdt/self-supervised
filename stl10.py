from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
from transforms import MultiSample, aug_transform


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.43, .42, .39), (.27, .26, .27))])


def test_transform():
    return T.Compose([
        T.Resize(70, interpolation=3), T.CenterCrop(64), base_transform()])


def loader_train(batch_size):
    t = MultiSample(aug_transform(64, base_transform))
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
