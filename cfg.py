import argparse
from torchvision import models


def get_cfg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--w_iter', type=int, default=1)
    parser.add_argument('--w_slice', type=int, default=1)

    parser.add_argument('--nce', action='store_true')
    parser.add_argument('--no_norm', dest='norm', action='store_false')
    parser.add_argument('--tau', type=float, default=.5)

    parser.add_argument('--linear_head', action='store_true')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--drop', type=int, nargs='*', default=[180])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')
    return parser.parse_args()
