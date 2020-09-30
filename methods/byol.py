from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import get_model, get_head
from .base import BaseMethod
from .norm_mse import norm_mse_loss


class BYOL(BaseMethod):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, cfg):
        """ init additional target and predictor networks """
        super().__init__(cfg)
        self.pred = nn.Sequential(
            nn.Linear(cfg.emb, cfg.head_size),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(),
            nn.Linear(cfg.head_size, cfg.emb),
        )
        self.model_t, _ = get_model(cfg.arch, cfg.dataset)
        self.head_t = get_head(self.out_size, cfg)
        for param in chain(self.model_t.parameters(), self.head_t.parameters()):
            param.requires_grad = False
        self.update_target(0)
        self.byol_tau = cfg.byol_tau
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def forward(self, samples):
        z = [self.pred(self.head(self.model(x))) for x in samples]
        with torch.no_grad():
            zt = [self.head_t(self.model_t(x)) for x in samples]

        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                loss += self.loss_f(z[i], zt[j]) + self.loss_f(z[j], zt[i])
        loss /= self.num_pairs
        return loss

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)
