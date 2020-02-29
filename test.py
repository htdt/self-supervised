import argparse
import torch
from model import get_model
from eval_lbfgs import eval_lbfgs
from eval_sgd import eval_sgd
from torchvision import models
import cifar10
import stl10
import tiny_in
DS = {'cifar10': cifar10, 'stl10': stl10, 'tiny_in': tiny_in}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--emb', type=int, default=128)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet50')
    parser.add_argument('--clf', type=str, default='sgd',
                        choices=['sgd', 'lbfgs'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'stl10', 'tiny_in'])
    parser.add_argument('--fname', type=str)
    cfg = parser.parse_args()

    model, out_size = get_model(cfg.arch, cfg.dataset)
    if cfg.fname is None:
        print('evaluating random model')
    else:
        checkpoint = torch.load(cfg.fname)
        model.load_state_dict(checkpoint['model'])

    loader_clf = DS[cfg.dataset].loader_clf()
    loader_test = DS[cfg.dataset].loader_test()

    if cfg.clf == 'sgd':
        acc = eval_sgd(model, out_size, loader_clf, loader_test, 500)
    elif cfg.clf == 'lbfgs':
        acc = eval_lbfgs(model, loader_clf, loader_test)
    print(acc)
