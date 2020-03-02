from abc import ABCMeta, abstractmethod
from functools import lru_cache
from torch.utils.data import DataLoader


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, bs_train, bs_clf=1000, bs_test=1000):
        self.bs_train = bs_train
        self.bs_clf = bs_clf
        self.bs_test = bs_test

    @abstractmethod
    def ds_train(self):
        raise NotImplementedError

    @abstractmethod
    def ds_clf(self):
        raise NotImplementedError

    @abstractmethod
    def ds_test(self):
        raise NotImplementedError

    @property
    @lru_cache()
    def train(self):
        return DataLoader(dataset=self.ds_train(),
                          batch_size=self.bs_train,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True)

    @property
    @lru_cache()
    def clf(self):
        return DataLoader(dataset=self.ds_clf(),
                          batch_size=self.bs_clf,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=True)

    @property
    @lru_cache()
    def test(self):
        return DataLoader(dataset=self.ds_test(),
                          batch_size=self.bs_test,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          drop_last=False)
