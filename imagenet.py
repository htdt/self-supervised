from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transforms import RandomApply2, MultiSample


def base_transform():
    return T.Compose([
        T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])


def crop224():
    return T.Compose([T.Resize(256, interpolation=3), T.CenterCrop(224)])


def test_transform():
    return T.Compose([crop224(), base_transform()])


def aug_transform(gp, jp, rp):
    return T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.RandomGrayscale(p=gp),
        T.RandomApply([T.ColorJitter(.4, .4, .4, .2)], p=jp),
        RandomApply2(T.RandomResizedCrop(224, scale=(.3, 1), interpolation=3),
                     crop224(), p=rp),
        base_transform()
    ])


def loader_train(batch_size):
    t = MultiSample(aug_transform(rp=.8, gp=.25, jp=.5))
    ts_train = ImageFolder(root='/imagenet/train', transform=t)
    return DataLoader(ts_train, batch_size=batch_size, shuffle=True,
                      num_workers=16, pin_memory=True, drop_last=True)


def loader_clf(batch_size=1000):
    ts_clf = ImageFolder(root='/imagenet/train', transform=test_transform())
    return DataLoader(ts_clf, batch_size=batch_size, shuffle=False,
                      num_workers=16, pin_memory=True, drop_last=True)


def loader_test(batch_size=1000):
    ts_test = ImageFolder(root='/imagenet/val/val',
                          transform=test_transform())
    return DataLoader(ts_test, batch_size=batch_size, shuffle=False,
                      num_workers=16)
