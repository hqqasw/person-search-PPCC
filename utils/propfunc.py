####################################
# File name: propfunc.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: Implement Label Propagation and Progressive Propagation via Competitive Consensus
####################################


import numpy as np


def softmax(x, T=1.0):
    expx = np.exp(x/T)
    exp_sum = np.sum(expx, axis=1)[:, np.newaxis]
    exp_sum[exp_sum == 0] = 1
    y = expx / exp_sum
    return y


def lp(W, Y0, steps=50):
    """
    lable propagation with linear difussion
    W: affinity matrix
    Y0: initial label matrix
    steps: number of iterations of propagation
    """
    num_instance, num_cast = Y0.shape
    Y = Y0.copy()
    W = softmax(W)
    for step in range(steps):
        Y = W.dot(Y)
    result = Y.T[:, num_cast:]
    return result


def ccpp(W, Y0, init_fratio=0.5, steps=5, temperature=0.03):
    """
    competitive consensus with progressive propagation
    W: affinity matrix
    Y0: initial label matrix
    init_fratio: initial frozen ration
    steps: number of iterations of propagation
    temperature: temperature of softmax
    """
    num_instance, num_cast = Y0.shape
    Y1 = np.zeros(Y0.shape)
    Y1[...] = Y0[...]
    frozen_mask = np.zeros(Y0.shape)
    for step in range(steps):
        fratio = init_fratio + step * ((1-init_fratio) / (steps-1))
        Y2 = np.zeros(Y1.shape)
        for i in range(num_cast):
            # for acceleration
            hot_num = int((Y1[:, i] != 0).sum())
            if hot_num == 0:
                continue
            hot_mask = np.zeros((num_instance, hot_num))
            hot_idxs = np.where(Y1[:, i] != 0)[0]
            hot_mask[hot_idxs, [x for x in range(hot_num)]] = Y1[hot_idxs, i]
            Y2[:, i] = W.dot(hot_mask).max(axis=1)
        Y2 = softmax(Y2, T=temperature)
        Y2 = Y1*frozen_mask + Y2*np.logical_not(frozen_mask)
        # progressive propagation
        max_value = np.max(Y2, axis=1)
        thr_idxs = np.argsort(max_value)[int(num_instance*(1-fratio)):]
        mask = np.zeros(Y1.shape)
        mask[thr_idxs] = 1
        frozen_mask = np.logical_or(frozen_mask, mask)
        Y1[...] = 0
        Y1[frozen_mask] = Y2[frozen_mask]
        Y1[Y1 < 0.001] = 0  # for acceleration
    result = (W.dot(Y0)+Y2).T[:, num_cast:]
    return result
