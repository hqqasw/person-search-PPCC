####################################
# File name: gpu_propfunc.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: GPU implement Label Propagation and Progressive Propagation via Competitive Consensus
####################################


import torch
import numpy as np


def gpu_softmax(x, T=1.0):
    expx = torch.exp(x/T)
    exp_sum = torch.sum(expx, dim=1).unsqueeze(dim=1)
    exp_sum[exp_sum == 0] = 1
    y = expx / exp_sum
    return y


def gpu_lp(W, Y0, gpu_id, steps=50):
    """
    lable propagation with linear difussion
    W: affinity matrix
    Y0: initial label matrix
    gpu_id: which gpu to use
    steps: number of iterations of propagation
    """
    num_instance, num_cast = Y0.shape
    gpu_Y = torch.tensor(Y0, device=gpu_id, dtype=torch.float)
    gpu_W = torch.tensor(W, device=gpu_id, dtype=torch.float)
    gpu_W = gpu_softmax(gpu_W, T=1.0)
    for step in range(steps):
        gpu_Y = torch.matmul(gpu_W, gpu_Y)
    Y = gpu_Y.detach().cpu().numpy()
    result = Y.T[:, num_cast:]
    return result


def gpu_ccpp(W, Y0, gpu_id, init_fratio=0.5, steps=5, temperature=0.03):
    """
    competitive consensus with progressive propagation
    W: affinity matrix
    Y0: initial label matrix
    gpu_id: which gpu to use
    init_fratio: initial frozen ration
    steps: number of iterations of propagation
    temperature: temperature of softmax
    """
    num_instance, num_cast = Y0.shape
    gpu_W = torch.tensor(W, device=gpu_id, dtype=torch.float)
    Y1 = torch.tensor(Y0, device=gpu_id, dtype=torch.float)
    frozen_mask = torch.zeros(num_instance, num_cast, device=gpu_id, dtype=torch.float)
    for step in range(steps):
        fratio = init_fratio + step * ((1-init_fratio) / (steps-1))
        Y2 = torch.zeros(num_instance, num_cast, device=gpu_id, dtype=torch.float)
        for i in range(num_cast):
            # for acceleration
            if (Y1[:, i] != 0).sum() == 0:
                continue
            gpu_diag = torch.diag(Y1[:, i])
            hot_mask = gpu_diag[:, Y1[:, i] != 0]
            Y2[:, i], _ = torch.matmul(gpu_W, hot_mask).max(dim=1)
        Y2 = gpu_softmax(Y2, T=temperature)
        Y2 = Y1*frozen_mask + Y2*(1 - frozen_mask)
        # progressive propagation
        max_value_vector, _ = torch.max(Y2, dim=1)
        thr_value = torch.sort(max_value_vector)[0][int(num_instance*(1-fratio))]
        mask = torch.zeros(num_instance, num_cast, device=gpu_id, dtype=torch.float)
        mask[max_value_vector > thr_value] = 1
        frozen_mask = frozen_mask + mask
        frozen_mask[frozen_mask > 0] = 1
        Y1 = torch.zeros(num_instance, num_cast, device=gpu_id, dtype=torch.float)
        Y1[frozen_mask > 0] = Y2[frozen_mask > 0]
        Y1[Y1 < 0.001] = 0  # for acceleration
    Y = Y2.detach().cpu().numpy()
    result = (W.dot(Y0)+Y).T[:, num_cast:]
    return result
