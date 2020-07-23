import torch.nn as nn
from torchvision import models
from dim_encoder import DIM32, DIM64
from whitening.ZCANorm import ZCANormSVDPI
from whitening.cholesky import Whitening2d


def get_head(out_size, cfg):
    x = []
    in_size = out_size
    for _ in range(cfg.head_layers - 1):
        x.append(nn.Linear(in_size, cfg.head_size))
        if cfg.add_bn:
            x.append(nn.BatchNorm1d(cfg.head_size))
        x.append(nn.ReLU())
        in_size = cfg.head_size
    x.append(nn.Linear(in_size, cfg.emb))
    if cfg.add_bn_last:
        x.append(nn.BatchNorm1d(cfg.emb))

    if cfg.w_mse:
        if cfg.method == "cholesky":
            x.append(Whitening2d(cfg.emb, eps=cfg.w_eps, track_running_stats=False))
        elif cfg.method == "zca":
            x.append(ZCANormSVDPI(cfg.emb, eps=cfg.w_eps))
        else:
            raise Exception("unknown method")
    return nn.Sequential(*x).cuda().train()


def get_model(arch, dataset):
    if arch == "DIM32":
        model, out_size = DIM32(), 1024
    elif arch == "DIM64":
        model, out_size = DIM64(), 4096
    else:
        model = getattr(models, arch)(pretrained=False)
        if dataset != "imagenet":
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        if dataset == "cifar10" or dataset == "cifar100":
            model.maxpool = nn.Identity()
        out_size = model.fc.in_features
        model.fc = nn.Identity()

    return nn.DataParallel(model).cuda().train(), out_size
