import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


if __name__ == '__main__':
    cpoint = torch.load('data/emb.pt')
    x_train = cpoint['x_train'].cuda()
    y_train = cpoint['y_train'].cuda()
    x_test = cpoint['x_test'].cuda()
    y_test = cpoint['y_test'].cuda()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--lr_start', type=float, default=1e-2)
    parser.add_argument('--lr_end', type=float, default=1e-6)
    parser.add_argument('--l2', type=float, default=5e-6)
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    clf = nn.Linear(2048, 1000)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(
        clf.parameters(), lr=cfg.lr_start, weight_decay=cfg.l2)

    gamma = (cfg.lr_end / cfg.lr_start) ** (1 / cfg.epoch)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in trange(cfg.epoch):
        loss_ep = []
        cut_last = (len(x_train) // cfg.bs) * cfg.bs
        perm = torch.randperm(len(x_train))[:cut_last].view(-1, cfg.bs)
        for idx in perm:
            optimizer.zero_grad()
            loss = criterion(clf(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        scheduler.step()

        if (ep + 1) % 5 == 0:
            clf.eval()
            with torch.no_grad():
                y_pred = clf(x_test)
            acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
            wandb.log({'acc': acc})
            clf.train()
