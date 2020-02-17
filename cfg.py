import argparse
from torchvision import models


def get_cfg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--whitening', action='store_true')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--drop', type=int, nargs='*', default=[250, 280])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'stl10', 'imagenet'])
    return parser.parse_args()
