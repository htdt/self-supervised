import argparse
import torch
import wandb
from dataset import get_loader_clf, get_loader_test
from model import get_model
from clf import eval_lbfgs, eval_sgd
from torchvision import models
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument(
        '--clf', type=str, choices=['sgd', 'lbfgs'], default='sgd')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'stl10', 'imagenet'])
    parser.add_argument('--fname', type=str)
    cfg = parser.parse_args()
    wandb.init(project="white_ss", config=cfg)

    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset == 'cifar10')
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    loader_clf = DS[cfg.dataset].loader_clf(aug=cfg.clf == 'sgd')
    loader_test = DS[cfg.dataset].loader_test()
    if cfg.clf == 'sgd':
        eval_sgd(model, head.in_features, loader_clf, loader_test, cfg.epoch)
    elif cfg.clf == 'lbfgs':
        eval_lbfgs(model, loader_clf, loader_test)
