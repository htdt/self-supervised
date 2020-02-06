import argparse
from tqdm import trange
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import wandb
from dataset import get_loader_clf, get_loader_test


def get_acc(model, loader_clf, loader_test, epoch, milestones, lr):
    model.eval()
    output_size = model.fc.in_features
    model.fc = nn.Identity()
    clf = nn.Linear(output_size, 10)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    criterion = nn.CrossEntropyLoss()

    x_test, y_test = [], []
    with torch.no_grad():
        for x, y in loader_test:
            x = x.cuda()
            x_test.append(model(x))
            y_test.append(y)
    x_test = torch.cat(x_test)
    y_test = torch.cat(y_test).cuda()

    for ep in trange(epoch):
        loss_ep = []
        for x, y in loader_clf:
            with torch.no_grad():
                x = model(x.cuda())
            y = y.cuda()
            optimizer.zero_grad()
            loss = criterion(clf(x), y)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())

        if (ep + 1) % 5 == 0:
            clf.eval()
            with torch.no_grad():
                y_pred = clf(x_test)
            acc = (y_pred.argmax(1) == y_test).float().mean().cpu().item()
            wandb.log({'acc': acc}, commit=False)
            clf.train()

        wandb.log({'loss': np.mean(loss_ep), 'ep': ep})
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument('--im_s1', type=float, default=1.15)
    parser.add_argument('--im_s2', type=float, default=2.1)
    parser.add_argument('--im_rp', type=float, default=.05)
    parser.add_argument('--im_gp', type=float, default=.1)
    parser.add_argument('--im_jp', type=float, default=.1)
    parser.add_argument('--fname', type=str, required=True)
    cfg = parser.parse_args()
    cfg.mode = 'clf_sgd'
    wandb.init(project="white_ss", config=cfg)

    cfgd = cfg.__dict__
    aug = {k[3:]: cfgd[k] for k in cfgd.keys() if k.startswith('im_')}
    loader_clf = get_loader_clf(aug=aug)
    loader_test = get_loader_test()

    model = getattr(models, cfg.arch)(num_classes=cfg.emb)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.cuda()
    checkpoint = torch.load(cfg.fname)
    model.load_state_dict(checkpoint['model'])

    m = [cfg.epoch - i * 20 for i in range(3, 0, -1)]
    get_acc(model, loader_clf, loader_test, cfg.epoch, m, 5e-3)
