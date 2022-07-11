"""
@Author: Conghao Wong
@Date: 2022-06-20 21:41:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-22 16:26:42
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import codes as C


class AgentArgs(C.args.BaseArgTable):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def K(self) -> int:
        """
        Number of multiple generations when evaluating.
        The number of trajectories predicted for one agent
        is calculated by `N = args.K * args.Kc`,
        where `Kc` is the number of style channels.
        """
        return self._get('K', 1, argtype='dynamic')

    @property
    def K_train(self) -> int:
        """
        Number of multiple generations when training.
        """
        return self._get('K_train', 1, argtype='static')

    @property
    def Kc(self) -> int:
        """
        Number of style channels in `Agent` model.
        """
        return self._get('Kc', 20, argtype='static')

    @property
    def key_points(self) -> str:
        """
        A list of key-time-steps to be predicted in the agent model.
        For example, `'0_6_11'`.
        """
        return self._get('key_points', '0_6_11', argtype='static')

    @property
    def depth(self) -> int:
        """
        Depth of the random contract id.
        """
        return self._get('depth', 16, argtype='static')

    @property
    def use_maps(self) -> int:
        """
        Controls if using the trajectory maps and the social maps in the model.
        DO NOT change this arg manually.
        """
        return self._get('use_maps', 1, argtype='static')

    @property
    def preprocess(self) -> str:
        """
        Controls if running any preprocess before model inference.
        Accept a 3-bit-like string value (like `'111'`):
        - the first bit: `MOVE` trajectories to (0, 0);
        - the second bit: re-`SCALE` trajectories;
        - the third bit: `ROTATE` trajectories.
        """
        return self._get('preprocess', '111', argtype='static')

    @property
    def T(self) -> str:
        """
        Type of transformations used when encoding or decoding
        trajectories.
        It could be:
        - `fft`: fast fourier transform
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._get('T', 'fft', argtype='static')

    @property
    def feature_dim(self) -> int:
        """
        Feature dimension used in most layers.
        """
        return self._get('feature_dim', 128, argtype='static')

    @property
    def metric(self) -> str:
        """
        Controls the metric used to save model weights when training.
        Accept either `'ade'` or `'fde'`.
        """
        return self._get('metric', 'fde', argtype='static')


class HandlerArgs(C.args.BaseArgTable):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def points(self) -> int:
        """
        Controls the number of keypoints accepted in the handler model.
        """
        return self._get('points', 1, argtype='static')

    @property
    def key_points(self) -> str:
        """
        A list of key-time-steps to be predicted in the handler model.
        For example, `'0_6_11'`.
        When setting it to `'null'`, it will start training by randomly sampling
        keypoints from all future moments.
        """
        return self._get('key_points', 'null', argtype='static')


class SilverballersArgs(C.args.BaseArgTable):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def loada(self) -> str:
        """
        Path for agent model.
        """
        return self._get('loada', 'null', argtype='dynamic')

    @property
    def loadb(self) -> str:
        """
        Path for handler model.
        """
        return self._get('loadb', 'null', argtype='dynamic')

    @property
    def K(self) -> int:
        """
        Number of multiple generations when evaluating.
        The number of trajectories outputed for one agent
        is calculated by `N = args.K * Kc`,
        where `Kc` is the number of style channels
        (given by args in the agent model).
        """
        return self._get('K', 1, argtype='dynamic')
