"""
@Author: Conghao Wong
@Date: 2022-06-20 16:24:29
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:00:10
@Description: Pre-processing methods.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf


def move(trajs: tf.Tensor,
         para_dict: dict[str, tf.Tensor],
         ref: int = -1,
         use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Move a specific point to (0, 0) according to the reference time step.
    Default reference time step is the last obsetvation step.

    :param trajs: observations, shape = `[(batch,) obs, 2]`
    :param ref: reference point, default is `-1`

    :return traj_moved: moved trajectories
    :return para_dict: a dict of used parameters
    """
    if use_new_para_dict:
        ref_point = trajs[:, ref, :] if len(trajs.shape) == 3\
            else trajs[ref, :]

        # shape is [batch, 1, 2] or [1, 2]
        ref_point = tf.expand_dims(ref_point, -2)
        para_dict['MOVE'] = ref_point

    else:
        ref_point = para_dict['MOVE']

    if len(trajs.shape) == 4:   # (batch, K, n, 2)
        ref_point = ref_point[:, tf.newaxis, :, :]

    traj_moved = trajs - ref_point

    return traj_moved, para_dict


def move_back(trajs: tf.Tensor,
              para_dict: dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Move trajectories back to their original positions.

    :param trajs: trajectories moved to (0, 0) with reference point, shape = `[(batch,) (K,) pred, 2]`
    :param para_dict: a dict of used parameters, which contains `'ref_point': tf.Tensor`

    :return traj_moved: moved trajectories
    """
    try:
        ref_point = para_dict['MOVE']  # shape = [(batch,) 1, 2]
        if len(ref_point.shape) == len(trajs.shape):
            traj_moved = trajs + ref_point
        else:   # [(batch,) K, pred, 2]
            traj_moved = trajs + tf.expand_dims(ref_point, -3)
        return traj_moved

    except:
        return trajs


def rotate(trajs: tf.Tensor,
           para_dict: dict[str, tf.Tensor],
           ref: int = 0,
           use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Rotate trajectories to the referce angle.

    :param trajs: observations, shape = `[(batch,) obs, 2]`
    :param ref: reference angle, default is `0`

    :return traj_rotated: moved trajectories
    :return para_dict: a dict of used parameters, `'rotate_angle': tf.Tensor`
    """
    if use_new_para_dict:
        vector_x = (trajs[:, -1, 0] - trajs[:, 0, 0]) if len(trajs.shape) == 3 else (
            trajs[-1, 0] - trajs[0, 0])  # shape is [batch] or []
        vector_y = (trajs[:, -1, 1] - trajs[:, 0, 1]) if len(trajs.shape) == 3 else (
            trajs[-1, 1] - trajs[0, 1])  # shape is [batch] or []

        main_angle = tf.atan((vector_y + 0.01)/(vector_x + 0.01))
        angle = ref - main_angle
        para_dict['ROTATE'] = angle

    else:
        angle = para_dict['ROTATE']

    rotate_matrix = tf.stack([
        [tf.cos(angle), tf.sin(angle)],
        [-tf.sin(angle), tf.cos(angle)]
    ])  # shape = [2, 2, batch] or [2, 2]

    if len(trajs.shape) == 3:
        # reshape to (batch, 2, 2)
        rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

    traj_rotated = trajs @ rotate_matrix

    return traj_rotated, para_dict


def rotate_back(trajs: tf.Tensor,
                para_dict: dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Rotate trajectories back to their original angles.

    :param trajs: trajectories, shape = `[(batch, ) pred, 2]`
    :param para_dict: a dict of used parameters, `'rotate_matrix': tf.Tensor`

    :return traj_rotated: rotated trajectories
    """
    angle = -1 * para_dict['ROTATE']

    # shape = (2, 2, batch)
    rotate_matrix = tf.stack([[tf.cos(angle), tf.sin(angle)],
                              [-tf.sin(angle), tf.cos(angle)]])

    S = tf.cast(trajs.shape, tf.int32)

    if len(S) >= 3:
        # traj shape = (batch, pred, 2)
        rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

    if len(S) == 4:
        # traj shape = (batch, K, pred, 2)
        trajs = tf.reshape(trajs, (S[0]*S[1], S[2], S[3]))
        rotate_matrix = tf.repeat(rotate_matrix, S[1], axis=0)

    traj_rotated = trajs @ rotate_matrix

    if len(S) == 4:
        traj_rotated = tf.reshape(traj_rotated, S)

    return traj_rotated


def scale(trajs: tf.Tensor,
          para_dict: dict[str, tf.Tensor],
          ref: float = 1,
          use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Scale trajectories' direction vector into (x, y), where |x| <= 1, |y| <= 1.
    Reference point when scale is the `last` observation point.

    :param trajs: input trajectories, shape = `[(batch,) obs, 2]`
    :param ref: reference length, default is `1`
    :return traj_scaled: scaled trajectories
    :return para_dict: a dict of used parameters, contains `scale:tf.Tensor`
    """
    change_flag = False
    if len(trajs.shape) == 2:
        trajs = tf.expand_dims(trajs, 0)    # change into [batch, obs, 2]
        change_flag = True

    x = trajs[:, :, 0]  # shape = [batch, obs]
    y = trajs[:, :, 1]

    if use_new_para_dict:
        scale = tf.linalg.norm(
            trajs[:, -1, :] - trajs[:, 0, :], axis=-1)  # [batch]
        scale = tf.maximum(0.05, scale)
        scale = tf.expand_dims(scale, -1)   # [batch, 1]
        para_dict['SCALE'] = scale

    else:
        scale = para_dict['SCALE']

    # shape = [batch, obs]
    new_x = (x - x[:, -1:]) / scale + x[:, -1:]
    new_y = (y - y[:, -1:]) / scale + y[:, -1:]

    traj_scaled = tf.stack([new_x, new_y])  # shape = [2, batch, obs]
    traj_scaled = tf.transpose(traj_scaled, [1, 2, 0])

    if change_flag:
        traj_scaled = traj_scaled[0, :, :]

    return traj_scaled, para_dict


def scale_back(trajs: tf.Tensor,
               para_dict: dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Scale trajectories back to their original.
    Reference point is the `first` prediction point.

    :param trajs: trajectories, shape = `[(batch,) (K,) pred, 2]`
    :param para_dict: a dict of used parameters, contains `scale:tf.Tensor`
    :return traj_scaled: scaled trajectories
    """
    original_dim = len(trajs.shape)
    if original_dim < 4:
        for repeat in range(4 - original_dim):
            # change into [batch, K, pred, 2]
            trajs = tf.expand_dims(trajs, -3)

    x = trajs[:, :, :, 0]   # [batch, K, pred]
    y = trajs[:, :, :, 1]

    scale = para_dict['SCALE']  # [batch, 1]
    scale = tf.expand_dims(scale, 1)    # [batch, 1, 1]

    # shape = [batch, K, obs]
    new_x = (x - tf.expand_dims(x[:, :, 0], -1)) * \
        scale + tf.expand_dims(x[:, :, 0], -1)
    new_y = (y - tf.expand_dims(y[:, :, 0], -1)) * \
        scale + tf.expand_dims(y[:, :, 0], -1)

    traj_scaled = tf.stack([new_x, new_y])  # [2, batch, K, pred]
    traj_scaled = tf.transpose(traj_scaled, [1, 2, 3, 0])

    if original_dim < 4:
        for repeat in range(4 - original_dim):
            traj_scaled = traj_scaled[0]

    return traj_scaled


def upSampling(trajs: tf.Tensor,
               para_dict: dict[str, tf.Tensor],
               sample_time: int,
               use_new_para_dict=True):

    if use_new_para_dict:
        para_dict['UPSAMPLING'] = sample_time
    else:
        sample_time = para_dict['UPSAMPLING']

    original_number = trajs.shape[-2]
    sample_number = sample_time * original_number

    if len(trajs.shape) == 3:   # (batch, n, 2)
        return tf.image.resize(trajs[:, :, :, tf.newaxis], [sample_number, 2])[:, :, :, 0], para_dict

    elif len(trajs.shape) == 4:   # (batch, K, n, 2)
        K = trajs.shape[1]
        results = []
        for k in range(K):
            results.append(tf.image.resize(
                trajs[:, k, :, :, tf.newaxis],
                [sample_number, 2])[:, :, :, 0])

        return tf.transpose(tf.stack(results), [1, 0, 2, 3]), para_dict


def upSampling_back(trajs: tf.Tensor,
                    para_dict: dict[str, tf.Tensor]):
    sample_time = para_dict['UPSAMPLING']
    sample_number = trajs.shape[-2]
    original_number = sample_number // sample_time
    original_index = tf.range(original_number) * sample_time

    return tf.gather(trajs, original_index, axis=-2)


def update(new: Union[tuple, list],
           old: Union[tuple, list]) -> tuple:

    if type(old) == list:
        old = tuple(old)
    if type(new) == list:
        new = tuple(new)

    if len(new) < len(old):
        return new + old[len(new):]
    else:
        return new
