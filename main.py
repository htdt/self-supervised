from time import time
from tqdm import trange
import argparse
import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import resnet18

from dataset import get_loaders
from clf import get_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--drop', type=int, nargs='*', default=[250, 280])
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--im0_s1', type=float, default=1.0)
    parser.add_argument('--im0_s2', type=float, default=1.3)
    parser.add_argument('--im0_rp', type=float, default=.5)
    parser.add_argument('--im0_gp', type=float, default=.2)
    parser.add_argument('--im0_jp', type=float, default=.5)

    parser.add_argument('--im1_s1', type=float, default=1.4)
    parser.add_argument('--im1_s2', type=float, default=2.0)
    parser.add_argument('--im1_rp', type=float, default=.9)
    parser.add_argument('--im1_gp', type=float, default=.5)
    parser.add_argument('--im1_jp', type=float, default=.8)

    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    cfgd = cfg.__dict__
    aug0 = {k[4:]: cfgd[k] for k in cfgd.keys() if k.startswith('im0')}
    aug1 = {k[4:]: cfgd[k] for k in cfgd.keys() if k.startswith('im1')}
    loader_train, loader_clf, loader_test = get_loaders(
        cfg.bs, aug0, aug1)

    model = resnet18(num_classes=cfg.emb)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.drop is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.drop)
    criterion = torch.nn.CrossEntropyLoss()
    target = torch.arange(cfg.bs).cuda()

    cudnn.benchmark = True
    t1 = time()
    for ep in trange(cfg.epoch):
        loss_ep = []
        for x, _ in loader_train:
            x0 = x[0].cuda(non_blocking=True)
            x1 = x[1].cuda(non_blocking=True)
            optimizer.zero_grad()
            x0, x1 = model(x0), model(x1)
            logits = x0 @ x1.t()
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        if cfg.drop is not None:
            scheduler.step()

    t2 = time()
    acc = get_acc(model, loader_clf, loader_test, True)
    t3 = time()
    wandb.log({
        'time_train': t2 - t1,
        'time_test': t3 - t2,
        'acc': acc,
    })
