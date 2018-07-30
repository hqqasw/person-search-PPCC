####################################
# File name: matching.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: Person search by label propagation
#              (both conventional label propagation and propagation via competitive consensus)
####################################


import os
import os.path as osp
import numpy as np
import argparse

from utils import read_meta, parse_label, read_across_movie_meta
from utils import get_topk, get_mAP, affmat2retdict, affmat2retlist
from utils import read_affmat_of_one_movie, read_affmat_across_movies
from utils import lp, ccpp
from utils import gpu_lp, gpu_ccpp


def run_lp(ct_affmat, tt_affmat, gpu_id):
    n_cast, n_instance = ct_affmat.shape
    n_sample = n_cast + n_instance
    W = np.zeros((n_sample, n_sample))
    W[:n_cast, n_cast:] = ct_affmat
    W[n_cast:, :n_cast] = ct_affmat.T
    W[n_cast:, n_cast:] = tt_affmat
    Y0 = np.zeros((n_sample, n_cast))
    for i in range(n_cast):
        Y0[i, i] = 1
    if gpu_id < 0:
        result = lp(W, Y0)
    else:
        result = gpu_lp(W, Y0, gpu_id=gpu_id)
    return result


def run_ccpp(ct_affmat, tt_affmat, gpu_id):
    n_cast, n_instance = ct_affmat.shape
    n_sample = n_cast + n_instance
    W = np.zeros((n_sample, n_sample))
    W[:n_cast, n_cast:] = ct_affmat
    W[n_cast:, :n_cast] = ct_affmat.T
    W[n_cast:, n_cast:] = tt_affmat
    Y0 = np.zeros((n_sample, n_cast))
    for i in range(n_cast):
        Y0[i, i] = 1
    if gpu_id < 0:
        result = ccpp(W, Y0)
    else:
        result = gpu_ccpp(W, Y0, gpu_id=gpu_id)
    return result


def run_in_movie(data_dir, subset, algorithm, temporal_link, gpu_id):
    affinity_dir = osp.join(data_dir, 'affinity', subset, 'in')
    list_file = osp.join(data_dir, 'meta', subset+'.json')
    mid_list, meta_info = read_meta(list_file)

    average_mAP = 0
    search_count = 0
    average_top1 = 0
    average_top3 = 0
    average_top5 = 0
    for i, mid in enumerate(mid_list):
        # read data
        tnum = meta_info[mid]['num_tracklet']
        pids = meta_info[mid]['pids']
        gt_list, gt_dict = parse_label(meta_info, mid)

        # read affinity matrix
        if temporal_link:
            link_type = 'max'
        else:
            link_type = 'mean'
        ct_affmat = read_affmat_of_one_movie(affinity_dir, mid, region='face', data_type='ct', link_type=link_type)
        tt_affmat = read_affmat_of_one_movie(affinity_dir, mid, region='body', data_type='tt', link_type=link_type)

        # run algorithm
        if algorithm == 'ppcc':
            result = run_ccpp(ct_affmat, tt_affmat, gpu_id)
        elif algorithm == 'lp':
            result = run_lp(ct_affmat, tt_affmat, gpu_id)
        else:
            raise ValueError('No such algrothm: {}'.format(algorithm))

        # parse results and get performance
        ret_dict = affmat2retdict(result, pids)
        ret_list = affmat2retlist(result, pids)
        mAP = get_mAP(gt_dict, ret_dict)
        topk = get_topk(gt_list, ret_list)
        average_mAP += mAP*len(pids)
        search_count += len(pids)
        max_k = len(topk)
        if max_k < 3:
            top3 = 1
        else:
            top3 = topk[2]
        if max_k < 5:
            top5 = 1
        else:
            top5 = topk[4]
        average_top1 += topk[0]
        average_top3 += top3
        average_top5 += top5

    # get average performance
    average_mAP = average_mAP / search_count
    average_top1 = average_top1 / len(mid_list)
    average_top3 = average_top3 / len(mid_list)
    average_top5 = average_top5 / len(mid_list)
    print(
        'Average mAP: {:.4f}\tAverage top1: {:.4f}\tAverage top3: {:.4f}\tAverage top5: {:.4f}'.format(
            average_mAP, average_top1, average_top3, average_top5))


def run_across_movie(data_dir, subset, algorithm, temporal_link, gpu_id):
    affinity_dir = osp.join(data_dir, 'affinity', subset, 'across')
    list_file = osp.join(data_dir, 'meta', 'across_{}.json'.format(subset))
    pids, gt_list, gt_dict = read_across_movie_meta(list_file)

    # read affinity matrix
    if temporal_link:
        link_type = 'max'
    else:
        link_type = 'mean'
    ct_affmat = read_affmat_across_movies(affinity_dir, region='face', data_type='ct', link_type=link_type)
    tt_affmat = read_affmat_across_movies(affinity_dir, region='body', data_type='tt', link_type=link_type)

    # run algorithm
    if algorithm == 'ppcc':
        result = run_ccpp(ct_affmat, tt_affmat, gpu_id)
    elif algorithm == 'lp':
        result = run_lp(ct_affmat, tt_affmat, gpu_id)
    else:
        raise ValueError('No such algrothm: {}'.format(algorithm))

    # parse results and get performance
    ret_dict = affmat2retdict(result, pids)
    ret_list = affmat2retlist(result, pids)
    mAP = get_mAP(gt_dict, ret_dict)
    topk = get_topk(gt_list, ret_list)

    print(
        'mAP: {:.4f}\ttop1: {:.4f}\ttop3: {:.4f}\ttop5: {:.4f}'.format(
            mAP, topk[0], topk[2], topk[4]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, choices=['test'], default='test')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--exp', choices=['in', 'across'], default='in')
    parser.add_argument('--gpu_id', help='set to -1 if you want to use CPU', type=int, default=0)
    parser.add_argument('--algorithm', choices=['lp', 'ppcc'], default='ppcc')
    parser.add_argument('--temporal_link', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.exp == 'in':
        run_in_movie(args.data_dir, args.subset, args.algorithm, args.temporal_link, args.gpu_id)
    else:
        run_across_movie(args.data_dir, args.subset, args.algorithm, args.temporal_link, args.gpu_id)
