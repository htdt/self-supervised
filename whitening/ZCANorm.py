import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from whitening.torch_utils import power_iteration_unstable, power_iteration_once


class ZCANormSVDunstable(nn.Module):
    def __init__(self, num_features, groups=4, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormSVDunstable, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer("running_mean", torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer(
                "running_subspace{}".format(i), torch.eye(length, length)
            )

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = (
                torch.mm(x, x.t()) / (N * H * W)
                + torch.eye(C, out=torch.empty_like(x)) * self.eps
            )

            assert C % G == 0
            length = int(C / G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                u, e, v = torch.svd(xxtj[i])
                ratio = torch.cumsum(e, 0) / e.sum()
                counter_i = 1
                for j in range(length):
                    if (
                        e[j] <= self.eps
                    ):  # ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                        # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                        counter_i = j + 1  # at least keep 99.99% energy
                        break
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    lambda_ij = e[j]
                    if lambda_ij < 0:
                        print("eigenvalues: ", e)
                        print(
                            "Error message: negative SVD lambda_ij {} vs SVD lambda_ij {}..".format(
                                lambda_ij, e[j]
                            )
                        )
                        break
                    eigenvector_ij = v[:, j][..., None]
                    subspace += torch.mm(
                        eigenvector_ij, torch.rsqrt(lambda_ij) * eigenvector_ij.t()
                    )
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__("running_subspace" + str(i))
                    running_subspace.data = (
                        1 - self.momentum
                    ) * running_subspace.data + self.momentum * subspace.data
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = x - self.running_mean
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__("running_subspace" + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        super(ZCANormSVDunstable, self)._load_from_state_dict(
            state_dict,
            prefix,
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ZCANormPIunstable(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormPIunstable, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_unstable.apply
        self.register_buffer("running_mean", torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer(
                "running_subspace{}".format(i), torch.eye(length, length)
            )
            for j in range(length):
                self.register_buffer(
                    "eigenvector{}-{}".format(i, j), torch.ones(length, 1)
                )

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = (
                torch.mm(x, x.t()) / (N * H * W)
                + torch.eye(C, out=torch.empty_like(x)) * self.eps
            )

            assert C % G == 0
            length = int(C / G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                subspace = torch.zeros_like(xxtj[i])
                for j in range(length):
                    # initialize eigenvector with random values
                    eigenvector_ij = self.__getattr__("eigenvector{}-{}".format(i, j))
                    v = l2normalize(torch.randn_like(eigenvector_ij))
                    eigenvector_ij.data = v.data

                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_current = torch.mm(
                        xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij
                    ) / torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if j == 0:
                        lambda_ij = lambda_current
                    elif lambda_ij < lambda_current or lambda_current < self.eps:
                        break
                    else:
                        lambda_ij = lambda_current
                    subspace += torch.mm(
                        eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t())
                    )
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(
                        xxtj[i], eigenvector_ij.mm(eigenvector_ij.t())
                    )

                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__("running_subspace" + str(i))
                    running_subspace.data = (
                        1 - self.momentum
                    ) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = x - self.running_mean
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__("running_subspace" + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        super(ZCANormPIunstable, self)._load_from_state_dict(
            state_dict,
            prefix,
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ZCANormSVDPI(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormSVDPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer("running_mean", torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer(
                "running_subspace{}".format(i), torch.eye(length, length)
            )
            for j in range(length):
                self.register_buffer(
                    "eigenvector{}-{}".format(i, j), torch.ones(length, 1)
                )

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = (
                torch.mm(x, x.t()) / (N * H * W)
                + torch.eye(C, out=torch.empty_like(x)) * self.eps
            )

            assert C % G == 0
            length = int(C / G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0) / e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            # print("{}/{} eigen-vectors selected".format(j + 1, length))
                            # print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__(
                            "eigenvector{}-{}".format(i, j)
                        )
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__("eigenvector{}-{}".format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_ij = torch.mm(
                        xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij
                    ) / torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        print("eigenvalues: ", e)
                        print(
                            "Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(
                                lambda_ij, e[j]
                            )
                        )
                        break
                    diff_ratio = (lambda_ij - e[j]).abs() / e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(
                        eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t())
                    )
                    xxtj[i] = xxtj[i] - torch.mm(
                        xxtj[i], eigenvector_ij.mm(eigenvector_ij.t())
                    )
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__("running_subspace" + str(i))
                    running_subspace.data = (
                        1 - self.momentum
                    ) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            # xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr.squeeze(2).squeeze(2)

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = x - self.running_mean
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__("running_subspace" + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            # x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x.squeeze(2).squeeze(2)

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        super(ZCANormSVDPI, self)._load_from_state_dict(
            state_dict,
            prefix,
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
