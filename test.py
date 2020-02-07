import argparse
import torch
import wandb
from dataset import get_loader_clf, get_loader_test
from model import get_model
from clf import eval_lbfgs, eval_sgd
from torchvision import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--whitening', action='store_true')
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument(
        '--clf', type=str, choices=['sgd', 'lbfgs'], default='sgd')
    parser.add_argument('--im_s1', type=float, default=1.15)
    parser.add_argument('--im_s2', type=float, default=2.1)
    parser.add_argument('--im_rp', type=float, default=.05)
    parser.add_argument('--im_gp', type=float, default=.1)
    parser.add_argument('--im_jp', type=float, default=.1)
    parser.add_argument('--fname', type=str, required=True)

    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)
    model, cur_size = get_model(cfg.arch, cfg.emb, cfg.whitening)
    checkpoint = torch.load(cfg.fname)
    model.load_state_dict(checkpoint['model'])

    if cfg.clf == 'sgd':
        cfgd = cfg.__dict__
        aug = {k[3:]: cfgd[k] for k in cfgd.keys() if k.startswith('im_')}
        loader_clf = get_loader_clf(aug=aug)
        loader_test = get_loader_test()
        eval_sgd(model, cur_size, loader_clf, loader_test, cfg.epoch)

    elif cfg.clf == 'lbfgs':
        eval_lbfgs(model, get_loader_clf(), get_loader_test())
