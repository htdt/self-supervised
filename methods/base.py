import torch.nn as nn
from model import get_model, get_head
from eval.sgd import eval_sgd
from eval.knn import eval_knn


class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model, self.out_size = get_model(cfg.arch, cfg.dataset)
        self.head = get_head(self.out_size, cfg)
        self.knn = cfg.knn
        self.num_pairs = cfg.num_samples * (cfg.num_samples - 1) // 2

    def forward(self, samples):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test):
        self.eval()
        acc_knn = eval_knn(self.model, self.out_size, ds_clf, ds_test, self.knn)
        acc_linear = eval_sgd(self.model, self.out_size, ds_clf, ds_test, 500)
        self.train()
        return acc_knn, acc_linear

    def step(self, progress):
        pass
