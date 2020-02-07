import argparse
from torchvision import models


def get_cfg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--whitening', action='store_true')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--emb', type=int, default=32)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--drop', type=int, nargs='*', default=[250, 280])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nce_t', type=float, default=1)
    parser.add_argument(
        '--arch', type=str, choices=dir(models), default='resnet18')

    parser.add_argument('--im0_s1', type=float, default=1.15)
    parser.add_argument('--im0_s2', type=float, default=1.4)
    parser.add_argument('--im0_rp', type=float, default=.5)
    parser.add_argument('--im0_gp', type=float, default=.25)
    parser.add_argument('--im0_jp', type=float, default=.5)

    parser.add_argument('--im1_s1', type=float, default=1.4)
    parser.add_argument('--im1_s2', type=float, default=2.1)
    parser.add_argument('--im1_rp', type=float, default=.9)
    parser.add_argument('--im1_gp', type=float, default=.25)
    parser.add_argument('--im1_jp', type=float, default=.5)

    return parser.parse_args()
