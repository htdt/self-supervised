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
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--emb', type=int, default=64)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--num_drop', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    loader_train, loader_clf, loader_test = get_loaders(cfg.bs)

    model = resnet18(num_classes=cfg.emb)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    m = [cfg.epoch - (i + 1) * 20 for i in reversed(range(cfg.num_drop))]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=m)
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
        scheduler.step()

    t2 = time()
    acc = get_acc(model, loader_clf, loader_test)
    t3 = time()
    wandb.log({
        'time_train': t2 - t1,
        'time_test': t3 - t2,
        'acc': acc,
    })
