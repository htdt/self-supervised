from os import path
import argparse
import torch
import wandb
from model import get_model
from eval_lbfgs import eval_lbfgs
from eval_sgd import eval_sgd
from torchvision import models
import cifar10
import stl10
import imagenet
DS = {'cifar10': cifar10, 'stl10': stl10, 'imagenet': imagenet}


def download_recent():
    api = wandb.Api()
    run = api.run("alexander/white_ss/9348jutn")
    flist = [f.name for f in run.files() if f.name.endswith('.pt')]
    flist.sort(key=lambda x: int(x[:-3]))
    run.file(flist[-1]).download(root='data')
    return path.join('data', flist[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--emb', type=int, default=128)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet50')
    parser.add_argument('--clf', type=str, default='sgd',
                        choices=['sgd', 'lbfgs'])
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'stl10', 'imagenet'])
    parser.add_argument('--fname', type=str)
    parser.add_argument('--download', action="store_true")
    cfg = parser.parse_args()
    if cfg.download:
        cfg.fname = download_recent()
    wandb.init(project="white_ss", config=cfg)

    model, head = get_model(cfg.arch, cfg.emb, cfg.dataset)
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    loader_clf = DS[cfg.dataset].loader_clf()
    loader_test = DS[cfg.dataset].loader_test()

    if cfg.clf == 'sgd':
        eval_sgd(model, head.module.in_features, loader_clf, loader_test)

    elif cfg.clf == 'lbfgs':
        acc = eval_lbfgs(model, loader_clf, loader_test)
        wandb.log({'acc': acc})
