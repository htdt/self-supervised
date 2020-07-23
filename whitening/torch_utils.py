import torch as th
# from torch.nn.functional import normalize
import torch.nn as nn
from time import sleep
import time
'''
This file contains all the custom pytorch operator.
'''


class power_iteration_unstable(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        vk_list = []
        vk_list.append(v_k)
        ctx.num_iter = num_iter
        for _ in range(int(ctx.num_iter)):
            v_k = M.mm(v_k)
            v_k /= th.norm(v_k).clamp(min=1.e-5)
            vk_list.append(v_k)

        ctx.save_for_backward(M, *vk_list)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M = ctx.saved_tensors[0]
        vk_list = ctx.saved_tensors[1:]
        dL_dvk1 = grad_output
        dL_dM = 0
        for i in range(1, ctx.num_iter + 1):
            v_k1 = vk_list[-i]
            v_k = vk_list[-i - 1]
            mid = calc_mid(M, v_k, v_k1, dL_dvk1)
            dL_dM += mid.mm(th.t(v_k))
            dL_dvk1 = M.mm(mid)
        return dL_dM, dL_dvk1


def calc_mid(M, v_k, v_k1, dL_dvk1):
    I = th.eye(M.shape[-1], out=th.empty_like(M))
    mid = (I - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k)).clamp(min=1.e-5)
    mid = mid.mm(dL_dvk1)
    return mid


class power_iteration_once(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = th.eye(M.shape[-1], out=th.empty_like(M))
        numerator = I - v_k.mm(th.t(v_k))
        denominator = th.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = th.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak

