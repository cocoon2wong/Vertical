"""
@Author: Conghao Wong
@Date: 2021-08-23 16:14:22
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-23 16:45:16
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import biplist
import os

BASE_DIR = './datasets'
SUBSETS_DIR = './datasets/subsets'

if __name__ == '__main__':
    subsets = {}

    subsets['eth'] = dict(
        dataset='eth',
        dataset_dir='./data/eth/univ',
        order=[1, 0],
        paras=[6, 25],
        video_path='./videos/eth.mp4',
        weights=[[
            [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
            [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
            [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
        ], 0.65, 225, 0.6, 160],
        scale=1,
    )

    subsets['hotel'] = dict(
        dataset='hotel',
        dataset_dir='./data/eth/hotel',
        order=[0, 1],
        paras=[10, 25],
        video_path='./videos/hotel.mp4',
        weights=[[
            [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
            [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
            [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
        ], 0.54, 470, 0.54, 300],
        scale=1,
    )

    subsets['zara1'] = dict(
        dataset='zara1',
        dataset_dir='./data/ucy/zara/zara01',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/zara1.mp4',
        weights=[-42.54748107, 580.5664891, 47.29369894, 3.196071003],
        scale=1,
    )

    subsets['zara2'] = dict(
        dataset='zara2',
        dataset_dir='./data/ucy/zara/zara02',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/zara2.mp4',
        weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
        scale=1,
    )

    subsets['univ'] = dict(
        dataset='univ',
        dataset_dir='./data/ucy/univ/students001',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/students003.mp4',
        weights=[-41.1428, 576, 48, 0],
        scale=1,
    )

    subsets['zara3'] = dict(
        dataset='zara3',
        dataset_dir='./data/ucy/zara/zara03',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/zara2.mp4',
        weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
        scale=1,
    )

    subsets['univ3'] = dict(
        dataset='univ3',
        dataset_dir='./data/ucy/univ/students003',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/students003.mp4',
        weights=[-41.1428, 576, 48, 0],
        scale=1,
    )

    subsets['unive'] = dict(
        dataset='unive',
        dataset_dir='./data/ucy/univ/uni_examples',
        order=[1, 0],
        paras=[10, 25],
        video_path='./videos/students003.mp4',
        weights=[-41.1428, 576, 48, 0],
        scale=1,
    )

    testsets = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

    for path in [BASE_DIR, SUBSETS_DIR]:
        if not os.path.exists(path):
            os.mkdir(path)

    for ds in testsets:
        train_sets = []
        test_sets = []
        val_sets = []

        for d in subsets.keys():
            if d == ds:
                test_sets.append(d)
                val_sets.append(d)
            else:
                train_sets.append(d)

        biplist.writePlist({'train': train_sets,
                            'test': test_sets,
                            'val': val_sets},
                           os.path.join(BASE_DIR, '{}.plist'.format(ds)),
                           binary=False)

    for key, value in subsets.items():
        biplist.writePlist(value,
                           os.path.join(SUBSETS_DIR, '{}.plist'.format(key)),
                           binary=False)
