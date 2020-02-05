import argparse
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_loader_clf, get_loader_test
from clf import get_acc
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument('--im_s1', type=float, default=1.15)
    parser.add_argument('--im_s2', type=float, default=2.1)
    parser.add_argument('--im_rp', type=float, default=.5)
    parser.add_argument('--im_gp', type=float, default=.25)
    parser.add_argument('--im_jp', type=float, default=.5)
    parser.add_argument('--fname', type=str, required=True)
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    cfgd = cfg.__dict__
    aug = {k[3:]: cfgd[k] for k in cfgd.keys() if k.startswith('im_')}
    loader_clf = get_loader_clf(aug)
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
