from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transforms import MultiSample, aug_transform


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.480, .448, .398), (.277, .269, .282))])


def loader_train(batch_size):
    t = MultiSample(aug_transform(64, base_transform))
    ts_train = ImageFolder(root='data/tiny-imagenet-200/train', transform=t)
    return DataLoader(ts_train, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_clf(batch_size=1000):
    ts_clf = ImageFolder(root='data/tiny-imagenet-200/train',
                         transform=base_transform())
    return DataLoader(ts_clf, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = ImageFolder(root='data/tiny-imagenet-200/test',
                          transform=base_transform())
    return DataLoader(ts_test, batch_size=batch_size, shuffle=False,
                      num_workers=8)
