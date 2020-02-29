import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import conv2d


class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True,
                 eps=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(
                [1, self.num_features, 1, 1]))
            self.register_buffer('running_variance',
                                 torch.eye(self.num_features))

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
            eye, torch.cholesky(f_cov_shrinked), upper=False)[0]
        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1)

        decorrelated = conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean)
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance)

        return decorrelated.squeeze(2).squeeze(2)

    def extra_repr(self):
        return 'features={}, eps={}, momentum={}'.format(
            self.num_features, self.eps, self.momentum)


def get_head(out_size, emb, linear=False, whitening=False, multi_gpu=False):
    h = nn.Linear(out_size, emb)
    if not linear:
        h = nn.Sequential(nn.Linear(out_size, out_size), nn.ReLU(), h)
    if whitening:
        h = nn.Sequential(h, Whitening2d(emb, track_running_stats=False))
    if multi_gpu:
        h = nn.DataParallel(h)
    return h.cuda().train()


def get_model(arch, dataset):
    model = getattr(models, arch)(pretrained=False)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if dataset == 'cifar10':
        model.maxpool = nn.Identity()
    out_size = model.fc.in_features
    model.fc = nn.Identity()
    return nn.DataParallel(model).cuda().train(), out_size
