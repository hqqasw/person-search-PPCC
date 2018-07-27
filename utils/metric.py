####################################
# File name: metric.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: mAP and topk recall
####################################


import numpy as np


def affmat2retlist(affmat, pid_list):
    """
    parse affinity matrix to resutl list
    """
    num_cast = affmat.shape[0]
    num_instance = affmat.shape[1]
    idxmat = np.argsort(-affmat, axis=0)
    ret_list = []
    for i in range(num_instance):
        ret_list.append([])
        for j in range(num_cast):
            ret_list[i].append(pid_list[idxmat[j, i]])
    return ret_list


def get_topk(gt_list, ret_list):
    max_k = len(ret_list[0])
    valid_idx = []
    for i, x in enumerate(gt_list):
        if x != 'others':
            valid_idx.append(i)
    valid_ret_list = [ret_list[x] for x in valid_idx]
    valid_gt_list = [gt_list[x] for x in valid_idx]
    valid_cnt = len(valid_idx)
    hit = 0.0
    topk = []
    for k in range(max_k):
        for i in range(valid_cnt):
            if valid_ret_list[i][k] == valid_gt_list[i]:
                hit += 1
        topk.append(hit / valid_cnt)
    return topk


def affmat2retdict(affmat, pid_list):
    """
    parse affinity matrix to resutl dict
    ret_dict:
        key: target person id
        value: list of candidates
    """
    num_cast = affmat.shape[0]
    num_instance = affmat.shape[1]
    index = np.argsort(-affmat, axis=1)
    ret_dict = {}
    for i in range(num_cast):
        ret_dict[pid_list[i]] = index[i].tolist()
    return ret_dict


def unique(ret_list):
    """
    remove duplicate candidates in the resutl list
    """
    unique_list = []
    unique_set = set()
    for x in ret_list:
        if x not in unique_set:
            unique_set.add(x)
            unique_list.append(x)
    return unique_list


def get_AP(gt_set, ret_list):
    hit = 0
    AP = 0.0
    unique_list = unique(ret_list)
    for k, x in enumerate(ret_list):
        if x in gt_set:
            hit += 1
            prec = hit / (k+1)
            AP += prec
    AP /= len(gt_set)
    return AP


def get_mAP(gt_dict, ret_dict):
    mAP = 0.0
    query_num = len(gt_dict.keys())
    for key, gt_set in gt_dict.items():
        if ret_dict.get(key) is None:
            AP = 0
        else:
            AP = get_AP(gt_set, ret_dict[key])
        mAP += AP
    mAP /= query_num
    return mAP
