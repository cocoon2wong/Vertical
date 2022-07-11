"""
@Author: Conghao Wong
@Date: 2022-06-20 10:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:20:44
@Description: Args for training and test trajectory prediction models.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import json
import os
import time
from typing import Any

from ..utils import dir_check

TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))


class BaseArgTable():
    """
    A set of args used for training or evaluating prediction models.
    """

    def __init__(self, terminal_args: list[str] = None) -> None:

        # args that load from the saved json file
        self._args_load: dict[str, Any] = {}

        # args that obtained from terminal
        self._args_runnning: dict[str, Any] = {}

        # args that set manually
        self._args_manually: dict[str, Any] = {}

        # a list that contains all args' names
        self._arg_list = [s for s in self.__dir__() if not s.startswith('_')]
        self._arg_list.sort()

        if terminal_args:
            self._load_from_terminal(terminal_args)

        if (l := self.load) != 'null':
            self._load_from_json(l)

    def _load_from_json(self, dir_path: str):
        try:
            arg_paths = [(p := os.path.join(dir_path, item)) for item in os.listdir(dir_path) if (
                item.endswith('args.json'))]

            with open(p, 'r') as f:
                json_dict = json.load(f)

            self._args_load = json_dict

        except:
            raise ValueError(
                'Failed to load args from path {}.'.format(dir_path))

    def _load_from_terminal(self, argv: list[str]):
        dic = {}

        index = 1
        while True:
            try:
                if argv[index].startswith('--'):
                    name = argv[index][2:]
                    value = argv[index+1]

                dic[name] = value
                index += 2

            except:
                break

        self._args_runnning = dic

    def _save_as_json(self, dir_path: str):
        json_path = os.path.join(dir_path, 'args.json')

        names = self._arg_list
        values = [getattr(self, s) for s in self._arg_list]

        with open(json_path, 'w+') as f:
            json.dump(dict(zip(names, values)), f,
                      separators=(',\n', ':'))

    def _get_args_by_index_and_name(self, index: int, name: str):
        if index == 0:
            dic = self._args_load
        elif index == 1:
            dic = self._args_runnning
        elif index == 99:
            dic = self._args_manually
        else:
            raise ValueError('Args index not exist.')

        return dic[name] if name in dic.keys() else None

    def _set(self, name: str, value: Any):
        """
        Set argument manually.
        """
        self._args_manually[name] = value

    def _get(self, name: str, default: Any, argtype: str):
        """
        Get arg by name

        :param name: name of the arg
        :param default: default value of the arg
        :param argtype: type of the arg, canbe
            - `'static'`
            - `'dynamic'`
            - ...
        """

        # arg dict index:
        # _args_load: 0
        # _args_running: 1
        # _args_manually: 99

        if argtype == 'static':
            order = [99, 0, 1]
        elif argtype == 'dynamic':
            order = [99, 1, 0]
        else:
            raise ValueError('Wrong arg type.')

        value = None
        for index in order:
            value = self._get_args_by_index_and_name(index, name)

            if value:
                break
            else:
                continue

        if not value:
            value = default

        value = type(default)(value)

        return value

    """
    Basic Model Args
    """
    @property
    def batch_size(self) -> int:
        """
        Batch size when implementation.
        """
        return self._get('batch_size', 5000, argtype='dynamic')

    @property
    def epochs(self) -> int:
        """
        Maximum training epochs.
        """
        return self._get('epochs', 500, argtype='static')

    @property
    def force_set(self) -> str:
        """
        Force test dataset. 
        Only works when evaluating when `test_mode` is `one`.
        """
        fs = self._get('force_set', 'null', argtype='dynamic')
        return fs

    @property
    def gpu(self) -> str:
        """
        Speed up training or test if you have at least one nvidia GPU. 
        If you have no GPUs or want to run the code on your CPU, 
        please set it to `-1`.
        """
        return self._get('gpu', '0', argtype='dynamic')

    @property
    def save_base_dir(self) -> str:
        """
        Base folder to save all running logs.
        """
        return self._get('save_base_dir', './logs', argtype='static')

    @property
    def save_model(self) -> int:
        """
        Controls if save the final model at the end of training.
        """
        return self._get('save_model', 1, argtype='static')

    @property
    def start_test_percent(self) -> float:
        """
        Set when to start validation during training.
        Range of this arg is `0 <= x <= 1`. 
        Validation will start at `epoch = args.epochs * args.start_test_percent`.
        """
        return self._get('start_test_percent', 0.0, argtype='static')

    @property
    def log_dir(self) -> str:
        """
        Folder to save training logs and models. If set to `null`,
        logs will save at `args.save_base_dir/current_model`.
        """
        log_dir_current = (TIME +
                           self.model_name +
                           self.model +
                           self.test_set)
        default_log_dir = os.path.join(dir_check(self.save_base_dir),
                                       log_dir_current)
        return self._get('log_dir', dir_check(default_log_dir), argtype='static')

    @property
    def load(self) -> str:
        """
        Folder to load model. If set to `null`,
        it will start training new models according to other args.
        """
        return self._get('load', 'null', argtype='dynamic')

    @property
    def model(self) -> str:
        """
        Model type used to train or test.
        """
        return self._get('model', 'none', argtype='static')

    @property
    def model_name(self) -> str:
        """
        Customized model name.
        """
        return self._get('model_name', 'model', argtype='static')

    @property
    def restore(self) -> str:
        """
        Path to restore the pre-trained weights before training.
        It will not restore any weights if `args.restore == 'null'`.
        """
        return self._get('restore', 'null', argtype='dynamic')

    @property
    def test_set(self) -> str:
        """
        Dataset used when training or evaluating.
        """
        return self._get('test_set', 'zara1', argtype='static')

    @property
    def test_step(self) -> int:
        """
        Epoch interval to run validation during training.
        """
        return self._get('test_step', 3, argtype='static')

    """
    Trajectory Prediction Args
    """
    @property
    def obs_frames(self) -> int:
        """
        Observation frames for prediction.
        """
        return self._get('obs_frames', 8, argtype='static')

    @property
    def pred_frames(self) -> int:
        """
        Prediction frames.
        """
        return self._get('pred_frames', 12, argtype='static')

    @property
    def draw_results(self) -> int:
        """
        Controls if draw visualized results on video frames.
        Make sure that you have put video files into `./videos`
        according to the specific name way.
        """
        return self._get('draw_results', 0, argtype='dynamic')

    @property
    def draw_distribution(self) -> int:
        """
        Conrtols if draw distributions of predictions instead of points.
        """
        return self._get('draw_distribution', 0, argtype='dynamic')

    @property
    def step(self) -> int:
        """
        Frame interval for sampling training data.
        """
        return self._get('step', 1, argtype='dynamic')

    @property
    def test_mode(self) -> str:
        """
        Test settings, canbe `'one'` or `'all'` or `'mix'`.
        When set it to `one`, it will test the model on the `args.force_set` only;
        When set it to `all`, it will test on each of the test dataset in `args.test_set`;
        When set it to `mix`, it will test on all test dataset in `args.test_set` together.
        """
        return self._get('test_mode', 'mix', argtype='dynamic')

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._get('lr', 0.001, argtype='static')

    @property
    def K(self) -> int:
        """
        Number of multiple generations when test.
        This arg only works for `Generative Models`.
        """
        return self._get('K', 20, argtype='dynamic')

    @property
    def K_train(self) -> int:
        """
        Number of multiple generations when training.
        This arg only works for `Generative Models`.
        """
        return self._get('K_train', 10, argtype='static')

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the context maps to model social
        and physical interactions in the model.
        """
        return self._get('use_maps', 1, argtype='static')

    @property
    def use_extra_maps(self) -> int:
        """
        Controls if uses the calculated trajectory maps or the given trajectory maps. 
        The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png`
        if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` 
        if set this argument to `1`.
        """
        return self._get('use_extra_maps', 0, argtype='dynamic')
