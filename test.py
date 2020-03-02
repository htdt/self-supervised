import argparse
import torch
from model import get_model
from eval_lbfgs import eval_lbfgs
from eval_sgd import eval_sgd
from torchvision import models
from datasets import get_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--emb', type=int, default=128)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet50')
    parser.add_argument('--clf', type=str, default='sgd',
                        choices=['sgd', 'lbfgs'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--fname', type=str)
    cfg = parser.parse_args()

    model, out_size = get_model(cfg.arch, cfg.dataset)
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    ds = get_ds(cfg.dataset)(None)
    if cfg.clf == 'sgd':
        acc = eval_sgd(model, out_size, ds.clf, ds.test, 500)
    elif cfg.clf == 'lbfgs':
        acc = eval_lbfgs(model, ds.clf, ds.test)
    print(acc)
