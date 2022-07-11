"""
@Author: Conghao Wong
@Date: 2022-06-21 09:41:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:54:50
@Description: Structures to manage trajectory datasets.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os


class VideoClip():
    """
    VideoClip
    ---------
    Base structure for controlling each video dataset.

    Properties
    -----------------
    ```python
    >>> self.name           # dataset name
    >>> self.dataset_dir    # dataset folder
    >>> self.order          # X, Y order
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.video_path     # video path
    >>> self.weights        # transfer weights from real scales to pixels
    >>> self.scale          # video scales
    ```
    """

    def __init__(self, name: str,
                 dataset_dir: str,
                 order: list[int],
                 paras: list[int],
                 video_path: str,
                 weights: list,
                 scale: float):

        self._name = name
        self._dataset_dir = dataset_dir
        self._order = order
        self._paras = paras
        self._video_path = video_path
        self._weights = weights
        self._scale = scale

    @staticmethod
    def get(dataset: str, root_dir='./datasets/subsets'):
        plist_path = os.path.join(root_dir, '{}.plist'.format(dataset))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(dataset))

        return VideoClip(**dic)

    @property
    def name(self):
        """
        Name of the video clip.
        """
        return self._name

    @property
    def dataset_dir(self):
        """
        Dataset folder, which contains a `*.txt` or `*.csv` 
        dataset file, and a scene image `reference.jpg`.
        """
        return self._dataset_dir

    @property
    def order(self):
        """
        order for coordinates, (x, y) -> `[0, 1]`, (y, x) -> `[1, 0]`
        """
        return self._order

    @property
    def paras(self):
        """
        [sample_step, frame_rate]
        """
        return self._paras

    @property
    def video_path(self):
        return self._video_path

    @property
    def weights(self):
        return self._weights

    @property
    def scale(self):
        return self._scale


class Dataset():
    """
    Dataset
    -------
    Manage a full trajectory prediction dataset.
    A dataset may contains several video clips.

    """

    def __init__(self, name: str, root_dir='./datasets'):
        plist_path = os.path.join(root_dir, '{}.plist'.format(name))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(name))

        self.train_sets : list[str] = dic['train']
        self.test_sets : list[str] = dic['test']
        self.val_sets : list[str] = dic['val']


def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.
    It is used to fix error when loading plist files through
    `biplist.readPlist()` in python 3.9 or newer.

    :param path: path of the plist file
    :return dat: a `dict` object loaded from the file
    """

    import plistlib
    import sys

    import biplist

    v = sys.version
    if int(v.split('.')[1]) >= 9:
        with open(path, 'rb') as f:
            dat = plistlib.load(f)
    else:
        dat = biplist.readPlist(path)

    return dat
