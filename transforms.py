import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Resize2:
    def __init__(self, s1, s2, cs):
        self.s1, self.s2 = s1, s2
        self.crop_fn = T.RandomCrop(cs)

    def __call__(self, x):
        ds = max(0, self.s2 - self.s1)
        size = self.s1 + ds * random.random()
        size = round(size * x.size[0])
        x = TF.resize(x, size)
        x = self.crop_fn(x)
        return x


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


class RandomApply2(object):
    def __init__(self, t1, t2, p):
        self.t1, self.t2, self.p = t1, t2, p

    def __call__(self, x):
        return self.t1(x) if random.random() < self.p else self.t2(x)
