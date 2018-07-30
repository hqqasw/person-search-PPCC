####################################
# File name: matching.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: Person search by visual matching
####################################


import os
import os.path as osp
import numpy as np
import argparse

from utils import read_meta, parse_label, read_across_movie_meta
from utils import read_affmat_of_one_movie, read_affmat_across_movies
from utils import get_topk, get_mAP, affmat2retdict, affmat2retlist


def run_in_movie(data_dir, subset, data_type, face_ratio):
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
        if data_type == 'face':
            affmat = read_affmat_of_one_movie(affinity_dir, mid, region='face', data_type='ct')
        elif data_type == 'body':
            affmat = read_affmat_of_one_movie(affinity_dir, mid, region='body', data_type='ct')
        else:
            face_affmat = read_affmat_of_one_movie(affinity_dir, mid, region='face', data_type='ct')
            body_affmat = read_affmat_of_one_movie(affinity_dir, mid, region='body', data_type='ct')
            if data_type == 'ave_fusion':
                affmat = face_ratio*face_affmat + (1-face_ratio)*body_affmat
            else:
                affmat = np.maximum(face_affmat, body_affmat)

        # parse results and get performance
        ret_dict = affmat2retdict(affmat, pids)
        ret_list = affmat2retlist(affmat, pids)
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


def run_across_movie(data_dir, subset, data_type, face_ratio):
    affinity_dir = osp.join(data_dir, 'affinity', subset, 'across')
    list_file = osp.join(data_dir, 'meta', 'across_{}.json'.format(subset))
    pids, gt_list, gt_dict = read_across_movie_meta(list_file)

    # read affinity matrix
    if data_type == 'face':
        affmat = read_affmat_across_movies(affinity_dir, region='face', data_type='ct')
    elif data_type == 'body':
        affmat = read_affmat_across_movies(affinity_dir, region='body', data_type='ct')
    else:
        face_affmat = read_affmat_across_movies(affinity_dir, region='face', data_type='ct')
        body_affmat = read_affmat_across_movies(affinity_dir, region='body', data_type='ct')
        if data_type == 'ave_fusion':
            affmat = face_ratio*face_affmat + (1-face_ratio)*body_affmat
        else:
            affmat = np.maximum(face_affmat, body_affmat)

    # parse results and get performance
    ret_dict = affmat2retdict(affmat, pids)
    ret_list = affmat2retlist(affmat, pids)
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
    parser.add_argument('--data_type', type=str, choices=['face', 'body', 'ave_fusion', 'max_fusion'], default='face')
    parser.add_argument('--face_ratio', type=float, default=0.8)
    args = parser.parse_args()
    print(args)

    if args.exp == 'in':
        run_in_movie(args.data_dir, args.subset, args.data_type, args.face_ratio)
    else:
        run_across_movie(args.data_dir, args.subset, args.data_type, args.face_ratio)
