from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transforms import RandomApply2, MultiSample


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])


def crop224():
    return T.Compose([T.Resize(256, interpolation=3), T.CenterCrop(224)])


def aug_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.RandomGrayscale(p=.25),
        T.RandomApply([T.ColorJitter(.4, .4, .4, .2)], p=.5),
        RandomApply2(T.RandomResizedCrop(224, scale=(.25, 1), interpolation=3),
                     crop224(), p=.8),
        base_transform()
    ])


def loader_train(batch_size):
    t = MultiSample(aug_transform())
    ts_train = ImageFolder(root='/imagenet/train', transform=t)
    return DataLoader(ts_train, batch_size=batch_size, shuffle=True,
                      num_workers=16, pin_memory=True, drop_last=True)


def loader_clf(batch_size=1000):
    ts_clf = ImageFolder(root='../imagenet224/train', transform=base_transform())
    return DataLoader(ts_clf, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = ImageFolder(root='../imagenet224/val', transform=base_transform())
    return DataLoader(ts_test, batch_size=batch_size, shuffle=False,
                      num_workers=8)
