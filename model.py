import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import conv2d
from dim_encoder import DIM32, DIM64
from ZCANorm import ZCANormSVDPI


class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.triangular_solve(
            eye, torch.cholesky(f_cov_shrinked), upper=False
        )[0]
        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )


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
