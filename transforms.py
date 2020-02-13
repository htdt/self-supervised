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
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(t(x) for t in self.transforms)


class WeightedRandomChoice(object):
    def __init__(self, transforms, probs):
        self.transforms = transforms
        self.probs = probs

    def __call__(self, x):
        t = random.choices(self.transforms, weights=self.probs)[0]
        return t(x)
