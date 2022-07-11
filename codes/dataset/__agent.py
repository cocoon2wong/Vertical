"""
@Author: Conghao Wong
@Date: 2022-06-21 09:26:56
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:53:14
@Description: Structure to manage training samples.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import copy

import numpy as np


class Agent():
    """
    Agent
    -----
    One agent manager contains these items for one specific agent:
    - historical trajectory: `traj`;
    - context map: `socialMap` and `trajMap`;
    - (future works): activity label;
    - (future works): agent category;
    - (future works): agent preference items

    Properties
    ----------
    ```python
    self.traj -> np.ndarray     # historical trajectory
    self.pred -> np.ndarray     # predicted (future) trajectory
    self.frames -> list[int]    # a list of frame index when this agent appeared
    self.frames_future -> list[int]     # agent's future frame index
    self.pred_linear -> np.ndarray  # agent's linear prediction
    self.groundtruth -> np.ndarray  # agent's future trajectory (when available)

    self.Map  -> np.ndarray   # agent's context map
    self.loss -> dict[str, np.ndarray]  # loss of agent's prediction
    ```

    Public Methods
    --------------
    ```python
    # copy this manager to a new address
    >>> self.copy() -> Agent

    # get neighbors' trajs -> list[np.ndarray]
    >>> self.get_neighbor_traj()

    # get neighbors' linear predictions
    >>> self.get_pred_traj_neighbor_linear() -> list[np.ndarray]
    ```
    """

    __version__ = 4.0

    _save_items = ['_traj', '_traj_future',
                   '_traj_pred', '_traj_pred_linear',
                   '_frames', '_frames_future',
                   'real2grid', '__version__',
                   'linear_predict',
                   'neighbor_number',
                   'neighbor_traj',
                   'neighbor_traj_linear_pred',
                   'obs_length', 'total_frame']
    
    def __init__(self):
        self._traj = []
        self._traj_future = []

        self._traj_pred = None
        self._traj_pred_linear = None

        self._map = None
        self.real2grid = None

        self._frames = []
        self._frames_future = []

        self.linear_predict = False
        self.obs_length = 0
        self.total_frame = 0

        self.neighbor_number = 0
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def copy(self):
        return copy.deepcopy(self)

    @property
    def traj(self) -> np.ndarray:
        """
        historical trajectory, shape = (obs, 2)
        """
        return self._traj

    @traj.setter
    def traj(self, value):
        self._traj = np.array(value).astype(np.float32)

    @property
    def pred(self) -> np.ndarray:
        """
        predicted trajectory, shape = (pred, 2)
        """
        return self._traj_pred

    @pred.setter
    def pred(self, value):
        self._traj_pred = np.array(value).astype(np.float32)

    @property
    def frames(self) -> list:
        """
        a list of frame index during observation and prediction time.
        shape = (obs + pred, 2)
        """
        return self._frames + self._frames_future

    @frames.setter
    def frames(self, value):
        self._frames = value if isinstance(value, list) else value.tolist()

    @property
    def frames_future(self) -> list:
        """
        a list of frame index during prediction time.
        shape = (pred, 2)
        """
        return self._frames_future

    @frames_future.setter
    def frames_future(self, value):
        if isinstance(value, list):
            self._frames_future = value
        elif isinstance(value, np.ndarray):
            self._frames_future = value.tolist()

    @property
    def pred_linear(self) -> np.ndarray:
        """
        linear prediction.
        shape = (pred, 2)
        """
        return self._traj_pred_linear

    @pred_linear.setter
    def pred_linear(self, value):
        self._traj_pred_linear = np.array(value).astype(np.float32)

    @property
    def groundtruth(self) -> np.ndarray:
        """
        ground truth future trajectory.
        shape = (pred, 2)
        """
        return self._traj_future

    @groundtruth.setter
    def groundtruth(self, value):
        self._traj_future = np.array(value).astype(np.float32)

    @property
    def Map(self) -> np.ndarray:
        """
        context map
        """
        return self._map

    def set_map(self, Map: np.ndarray, paras: np.ndarray):
        self._map = Map
        self.real2grid = paras

    def zip_data(self) -> dict[str, object]:
        zipped = {}
        for item in self._save_items:
            zipped[item] = getattr(self, item)
        return zipped

    def load_data(self, zipped_data: dict[str, object]):
        for item in self._save_items:
            if not item in zipped_data.keys():
                continue
            else:
                setattr(self, item, zipped_data[item])
        return self

    def init_data(self, target_traj, 
                  neighbors_traj,
                  frames, start_frame,
                  obs_frame, end_frame,
                  frame_step=1,
                  add_noise=False,
                  linear_predict=True):
        """
        Make one training data.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        Length (time steps) of `target_traj` and `neighbors_traj`
        are `(end_frame - start_frame) // frame_step`.
        """

        self.linear_predict = linear_predict

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        # Trajectory
        whole_traj = target_traj
        frames_current = frames

        # data strengthen: noise
        if add_noise:
            noise_curr = np.random.normal(0, 0.1, size=self.traj.shape)
            whole_traj += noise_curr

        self.frames = frames_current[:self.obs_length]
        self.traj = whole_traj[:self.obs_length]
        self.groundtruth = whole_traj[self.obs_length:]
        self.frames_future = frames_current[self.obs_length:]

        if linear_predict:
            self.pred_linear = predict_linear_for_person(
                self.traj, time_pred=self.total_frame)[self.obs_length:]

        # Neighbor info
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []
        for neighbor_traj in neighbors_traj:
            if neighbor_traj.max() >= 5000:
                available_index = np.where(neighbor_traj.T[0] <= 5000)[0]
                neighbor_traj[:available_index[0],
                              :] = neighbor_traj[available_index[0]]
                neighbor_traj[available_index[-1]:,
                              :] = neighbor_traj[available_index[-1]]
            self.neighbor_traj.append(neighbor_traj)

            if linear_predict:
                pred = predict_linear_for_person(neighbor_traj, time_pred=self.total_frame)[
                    self.obs_length:]
                self.neighbor_traj_linear_pred.append(pred)

        self.neighbor_number = len(neighbors_traj)
        return self

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def clear_all_neighbor_info(self):
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def get_pred_traj_neighbor_linear(self) -> list:
        return self.neighbor_traj_linear_pred


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def __predict_linear(x, y, x_p, diff_weights=0):
    if diff_weights == 0:
        P = np.diag(np.ones(shape=[x.shape[0]]))
    else:
        P = np.diag(softmax([(i+1)**diff_weights for i in range(x.shape[0])]))

    A = np.stack([np.ones_like(x), x]).T
    A_p = np.stack([np.ones_like(x_p), x_p]).T
    Y = y.T
    B = np.matmul(np.matmul(np.matmul(np.linalg.inv(
        np.matmul(np.matmul(A.T, P), A)), A.T), P), Y)
    Y_p = np.matmul(A_p, B)
    return Y_p, B


def predict_linear_for_person(position, time_pred, different_weights=0.95) -> np.ndarray:
    time_obv = position.shape[0]
    t = np.arange(time_obv)
    t_p = np.arange(time_pred)
    x = position.T[0]
    y = position.T[1]

    x_p, _ = __predict_linear(t, x, t_p, diff_weights=different_weights)
    y_p, _ = __predict_linear(t, y, t_p, diff_weights=different_weights)

    return np.stack([x_p, y_p]).T
