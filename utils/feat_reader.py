####################################
# File name: feat_reader.py
# Author: Qingqiu Huang
# Date created: 18/7/2018
# Date last modified: 18/7/2018
# Python Version: 3.6
# Description: Read meta data and affinity matrix
####################################


import json
import os
import os.path as osp
import numpy as np


# ********** meta data and labels reader **********
def read_meta(meta_file):
    with open(meta_file) as f:
        meta = json.load(f)
    mid_list = meta['movie']
    meta_info = meta['info']
    return mid_list, meta_info


def parse_label(meta_info, mid):
    """
    label_list:
        index: tracklet idx
        value: label
    label_dict:
        key: pid
        value: set of tracklet idx
    """
    label_list = meta_info[mid]['labels']
    pids = meta_info[mid]['pids']
    label_dict = {}
    for i, label in enumerate(label_list):
        if label_dict.get(label) is None:
            label_dict[label] = set()
        label_dict[label].add(i)
    return label_list, label_dict


def read_across_movie_meta(meta_file):
    """
    label_list:
        index: tracklet idx
        value: label
    label_dict:
        key: pid
        value: set of tracklet idx
    """
    with open(meta_file) as f:
        meta = json.load(f)
    label_list = []
    label_dict = {}
    pid_idx_map = {}
    for i, sample in enumerate(meta):
        idx = sample['plabel']
        pid = sample['pid']
        label_list.append(pid)
        pid_idx_map[idx] = pid
        if label_dict.get(pid) is None:
            label_dict[pid] = set()
        label_dict[pid].add(i)
    pid_num = len(pid_idx_map.keys())
    pids = [pid_idx_map[i] for i in range(pid_num)]
    return pids, label_list, label_dict


# ********** features reader **********
def read_feat_of_one_movie(feat_dir, mid, region, data_type):
    """
    region: face or body
    data_type: cast or tracklet
    """
    cache_file = osp.join(feat_dir, mid, '{}_{}_feat.npy'.format(data_type, region))
    if osp.isfile(cache_file):
        feat_mat = np.load(cache_file)
    else:
        raise IOError('No such feature cache! {}'.format(cache_file))
    return feat_mat


def read_feat_across_movies(feat_dir, region, data_type):
    """
    region: face or body
    data_type: cast or tracklet
    """
    cache_file = osp.join(feat_dir, '{}_{}_feat.npy'.format(data_type, region))
    if osp.isfile(cache_file):
        feat_mat = np.load(cache_file)
    else:
        raise IOError('No such feature cache! {}'.format(cache_file))
    return feat_mat


# ********** affinity reader **********
def read_affmat_of_one_movie(affinity_dir, mid, region, data_type, link_type='mean'):
    """
    region: face or body
    data_type: cast-tracklet or tracklet-tracklet
    link_type: max or mean
    """
    cache_file = osp.join(affinity_dir, mid, '{}_{}_{}_affmat.npy'.format(region, link_type, data_type))
    if osp.isfile(cache_file):
        affmat = np.load(cache_file)
    else:
        raise IOError('No such affinity matrix cache! {}'.format(cache_file))
    return affmat


def read_affmat_across_movies(affinity_dir, region, data_type, link_type='mean'):
    """
    region: face or body
    data_type: cast-tracklet or tracklet-tracklet
    link_type: max or mean
    """
    cache_file = osp.join(affinity_dir, '{}_{}_{}_affmat.npy'.format(region, link_type, data_type))
    if osp.isfile(cache_file):
        affmat = np.load(cache_file)
    else:
        raise IOError('No such affinity matrix cache! {}'.format(cache_file))
    return affmat
