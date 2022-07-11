"""
@Author: Conghao Wong
@Date: 2022-06-20 16:14:03
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:01:46
@Description: Base trajectory prediction model class.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re

import tensorflow as tf

from ..args import BaseArgTable
from . import __preprocess as preprocess

MOVE = 'MOVE'
ROTATE = 'ROTATE'
SCALE = 'SCALE'
UPSAMPLING = 'UPSAMPLING'


class Model(tf.keras.Model):
    """
    Model
    -----

    Usage
    -----
    When training or test new models, please subclass this class, and clarify
    model layers used in your model.
    ```python
    class MyModel(Model):
        def __init__(self, Args, structure, *args, **kwargs):
            super().__init__(Args, structure, *args, **kwargs)

            self.fc = tf.keras.layers.Dense(64, tf.nn.relu)
            self.fc1 = tf.keras.layers.Dense(2)
    ```

    Then define your model's pipeline in `call` method:
    ```python
        def call(self, inputs, training=None, mask=None):
            y = self.fc(inputs)
            return self.fc1(y)
    ```

    Public Methods
    --------------
    ```python
    # forward model with pre-process and post-process
    (method) forward: (self: Model, model_inputs: list[Tensor], training=None, *args, **kwargs) -> list[Tensor]

    # Pre/Post-processes
    (method) pre_process: (self: Model, tensors: list[Tensor], training=None, use_new_para_dict=True, *args, **kwargs) -> list[Tensor]
    (method) post_process: (self: Model, outputs: list[Tensor], training=None, *args, **kwargs) -> list[Tensor]
    ```
    """

    def __init__(self, Args: BaseArgTable,
                 structure=None,
                 *args, **kwargs):

        super().__init__()
        self.args = Args
        self.structure = structure

        # preprocess
        self._preprocess_list = []
        self._preprocess_para = {MOVE: -1,
                                 ROTATE: 0,
                                 SCALE: 1,
                                 UPSAMPLING: 4}

        self._preprocess_variables = {}

    def call(self, inputs,
             training=None,
             *args, **kwargs):

        raise NotImplementedError(self)

    def forward(self, inputs: list[tf.Tensor],
                training=None) -> list[tf.Tensor]:
        """
        Run a forward implementation.

        :param inputs: input tensor (or a `list` of tensors)
        :param training: config if running as training or test mode
        :return output: model's output. type=`list[tf.Tensor]`
        """

        inputs_processed = self.pre_process(inputs, training)

        # use `self.call()` to debug
        outputs = self(inputs_processed, training=training)

        # make sure the output is a list or a tuple
        if not type(outputs) in [list, tuple]:
            outputs = [outputs]

        return self.post_process(outputs, training,
                                 inputs=inputs)

    def set_preprocess(self, *args):
        """
        Set pre-process methods used before training.

        args: pre-process methods.
            - Move positions on the observation step to (0, 0):
                args in `['Move', ...]`

            - Re-scale observations:
                args in `['Scale', ...]`

            - Rotate observations:
                args in `['Rotate', ...]`
        """
        self._preprocess_list = []
        for item in args:
            if not issubclass(type(item), str):
                raise TypeError

            if re.match('.*[Mm][Oo][Vv][Ee].*', item):
                self._preprocess_list.append(MOVE)

            elif re.match('.*[Rr][Oo][Tt].*', item):
                self._preprocess_list.append(ROTATE)

            elif re.match('.*[Ss][Cc][Aa].*', item):
                self._preprocess_list.append(SCALE)

            elif re.match('.*[Uu][Pp].*[Ss][Aa][Mm].*', item):
                self._preprocess_list.append(UPSAMPLING)

    def set_preprocess_parameters(self, **kwargs):
        for item in kwargs.keys():
            if not issubclass(type(item), str):
                raise TypeError

            if re.match('.*[Mm][Oo][Vv][Ee].*', item):
                self._preprocess_para[MOVE] = kwargs[item]

            elif re.match('.*[Rr][Oo][Tt].*', item):
                self._preprocess_para[ROTATE] = kwargs[item]

            elif re.match('.*[Ss][Cc][Aa].*', item):
                self._preprocess_para[SCALE] = kwargs[item]

            elif re.match('.*[Uu][Pp].*[Ss][Aa][Mm].*', item):
                self._preprocess_para[UPSAMPLING] = kwargs[item]

    def pre_process(self, tensors: list[tf.Tensor],
                    training=None,
                    use_new_para_dict=True,
                    *args, **kwargs) -> list[tf.Tensor]:

        trajs = tensors[0]
        items = [MOVE, ROTATE, SCALE, UPSAMPLING]
        funcs = [preprocess.move, preprocess.rotate,
                 preprocess.scale, preprocess.upSampling]

        for item, func in zip(items, funcs):
            if item in self._preprocess_list:
                trajs, self._preprocess_variables = func(
                    trajs, self._preprocess_variables,
                    self._preprocess_para[item],
                    use_new_para_dict)

        return preprocess.update((trajs,), tensors)

    def post_process(self, outputs: list[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> list[tf.Tensor]:

        trajs = outputs[0]
        items = [MOVE, ROTATE, SCALE, UPSAMPLING]
        funcs = [preprocess.move_back, preprocess.rotate_back,
                 preprocess.scale_back, preprocess.upSampling_back]

        for item, func in zip(items[::-1], funcs[::-1]):
            if item in self._preprocess_list:
                trajs = func(trajs, self._preprocess_variables)

        return preprocess.update((trajs,), outputs)
