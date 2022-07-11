"""
@Author: Conghao Wong
@Date: 2022-06-21 10:44:39
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:58:51
@Description: Structures to manage all training samples in one video clip.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from .__agent import Agent


class Trajectory():
    """
    Entire Trajectory
    -----------------
    Manage one agent's entire trajectory in datasets.

    Properties
    ----------
    ```python
    >>> self.agent_index
    >>> self.traj
    >>> self.neighbors
    >>> self.frames
    >>> self.start_frame
    >>> self.end_frame
    ```
    """

    def __init__(self, agent_index: int,
                 trajectory: np.ndarray,
                 neighbors: list[list[int]],
                 frames: list[int],
                 init_position: float):
        """
        init

        :param agent_index: index of the trajectory
        :param neighbors: a list of lists that contain agents' ids \
            who appear in each frames. \
            index are frame indexes.
        :param trajectory: target trajectory, \
            shape = `(all_frames, 2)`.
        :param frames: a list of frame ids, \
            shaoe = `(all_frames)`.
        :param init_position: default position that indicates \
            agent has gone out of the scene.
        """

        self._agent_index = agent_index
        self._traj = trajectory  # matrix[:, agent_index, :]
        self._neighbors = neighbors
        self._frames = frames

        base = self.traj.T[0]
        diff = base[:-1] - base[1:]

        appear = np.where(diff > init_position/2)[0]
        # disappear in next step
        disappear = np.where(diff < -init_position/2)[0]

        self._start_frame = appear[0] + 1 if len(appear) else 0
        self._end_frame = disappear[0] + 1 if len(disappear) else len(base)

    @property
    def agent_index(self):
        return self._agent_index

    @property
    def traj(self):
        """
        Trajectory, shape = `(frames, 2)`
        """
        return self._traj

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def frames(self):
        """
        frame id that the trajectory appears.
        """
        return self._frames

    @property
    def start_frame(self):
        """
        index of the first observed frame
        """
        return self._start_frame

    @property
    def end_frame(self):
        """
        index of the last observed frame
        """
        return self._end_frame

    def sample(self, start_frame, obs_frame, end_frame,
               matrix,
               frame_step=1,
               max_neighbor=15,
               add_noise=False) -> Agent:
        """
        Sample training data from the trajectory.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        """
        neighbors = self.neighbors[obs_frame - frame_step]

        if len(neighbors) > max_neighbor + 1:
            nei_pos = matrix[obs_frame - frame_step, neighbors, :]
            tar_pos = self.traj[obs_frame - frame_step, np.newaxis, :]
            dis = calculate_length(nei_pos - tar_pos)
            neighbors = neighbors[np.argsort(dis)[1:max_neighbor+1]]

        nei_traj = matrix[start_frame:end_frame:frame_step, neighbors, :]
        nei_traj = np.transpose(nei_traj, [1, 0, 2])
        tar_traj = self.traj[start_frame:end_frame:frame_step, :]

        return Agent().init_data(target_traj=tar_traj,
                                 neighbors_traj=nei_traj,
                                 frames=self.frames[start_frame:end_frame:frame_step],
                                 start_frame=start_frame,
                                 obs_frame=obs_frame,
                                 end_frame=end_frame,
                                 frame_step=frame_step,
                                 add_noise=add_noise)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)
