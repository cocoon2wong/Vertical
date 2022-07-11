"""
@Author: Conghao Wong
@Date: 2021-08-23 16:24:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-23 16:48:22
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import biplist
import os

BASE_DIR = './datasets'
SUBSETS_DIR = './datasets/subsets'

if __name__ == '__main__':
    set_index = {'quad':   [[0, 1, 2, 3], 100.0],
                 'little':   [[0, 1, 2, 3], 100.0],
                 'deathCircle':   [[0, 1, 2, 3, 4], 100.0],
                 'hyang':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 100.0],
                 'nexus':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100.0],
                 'coupa':   [[0, 1, 2, 3], 100.0],
                 'bookstore':   [[0, 1, 2, 3, 4, 5, 6], 100.0],
                 'gates':   [[0, 1, 2, 3, 4, 5, 6, 7, 8], 100.0]}

    subsets = {}
    for base_set in set_index:
        for index in set_index[base_set][0]:
            subsets['{}{}'.format(base_set, index)] = dict(
                dataset='{}{}'.format(base_set, index),
                dataset_dir='./data/sdd/{}/video{}'.format(
                    base_set, index),
                order=[1, 0],
                paras=[1, 30],
                video_path='./videos/sdd_{}_{}.mov'.format(
                    base_set, index),
                weights=[set_index[base_set][1], 0.0,
                         set_index[base_set][1], 0.0],
                scale=2,
            )

    all_test_sets = ['hyang7',
                     'hyang11',
                     'bookstore6',
                     'nexus3',
                     'deathCircle4',
                     'hyang6',
                     'hyang3',
                     'little1',
                     'hyang13',
                     'gates8',
                     'gates7',
                     'hyang2']

    all_val_sets = ['nexus7',
                    'coupa1',
                    'gates4',
                    'little2',
                    'bookstore3',
                    'little3',
                    'nexus4',
                    'hyang4',
                    'gates3',
                    'quad2',
                    'gates1',
                    'hyang9']

    for path in [BASE_DIR, SUBSETS_DIR]:
        if not os.path.exists(path):
            os.mkdir(path)

    train_sets = []
    test_sets = []
    val_sets = []

    for d, dic in subsets.items():
        if d in all_test_sets:
            test_sets.append(d)
        elif d in all_val_sets:
            val_sets.append(d)
        else:
            train_sets.append(d)

    biplist.writePlist({'train': train_sets,
                        'test': test_sets,
                        'val': val_sets},
                       os.path.join(BASE_DIR, 'sdd.plist'),
                       binary=False)

    for key, value in subsets.items():
        biplist.writePlist(value,
                           os.path.join(SUBSETS_DIR, '{}.plist'.format(key)),
                           binary=False)
