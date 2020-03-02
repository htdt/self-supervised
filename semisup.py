import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import wandb
from datasets.stl10 import base_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--epoch_head', type=int, default=200)
    parser.add_argument('--fname', type=str)
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    model, out_size = get_model('resnet34', 'stl10')
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    head = nn.Sequential(
        nn.Linear(out_size, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, 10)).cuda()

    aug_t = T.Compose([
        T.RandomCrop(64),
        T.RandomHorizontalFlip(),
        base_transform()
    ])
    test_t = T.Compose([T.TenCrop(64), T.Lambda(
        lambda crops: torch.stack([base_transform()(c) for c in crops]))])

    ts_train = STL10(root='./data', split='train', download=True,
                     transform=aug_t)
    dl_train = DataLoader(ts_train, batch_size=1000, shuffle=True,
                          num_workers=8, pin_memory=True)
    ts_test = STL10(root='./data', split='test', download=True,
                    transform=test_t)
    dl_test = DataLoader(ts_test, batch_size=100, num_workers=8)

    def test():
        head.eval(), model.eval()
        acc = []
        for x, y in dl_test:
            bs, ncrops, c, h, w = x.shape
            x = x.cuda()
            x = x.view(bs * ncrops, c, h, w)
            with torch.no_grad():
                pred = head(model(x))
            pred = pred.view(bs, ncrops, 10).mean(1)
            acc.append((pred.argmax(1).cpu() == y).float().mean().item())
        return np.mean(acc)

    lr_start, lr_end = 1e-2, 1e-4
    gamma = (lr_end / lr_start) ** (1 / cfg.epoch_head)
    optimizer = optim.Adam(head.parameters(), lr=lr_start, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    model.eval(), head.train()
    for _ in trange(cfg.epoch_head):
        for x, y in dl_train:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                x = model(x)
            criterion(head(x), y).backward()
            optimizer.step()
        scheduler.step()
    wandb.log({'acc_head': test()})

    param = list(model.parameters()) + list(head.parameters())
    optimizer = optim.Adam(param, lr=1e-4, weight_decay=1e-4)

    model.train(), head.train()
    for ep in trange(cfg.epoch):
        loss_ep = []
        for x, y in dl_train:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            loss = criterion(head(model(x)), y)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
        if (ep + 1) % 20 == 0:
            wandb.log({'acc': test()}, commit=False)
            model.train(), head.train()
        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
