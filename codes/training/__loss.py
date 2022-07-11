"""
@Author: Conghao Wong
@Date: 2022-06-20 19:34:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:51:33
@Description: Basic loss functions used in trajectory prediction models.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re
from typing import Any, Union

import tensorflow as tf


def apply(loss_list: list[Union[str, Any]],
          model_outputs: list[tf.Tensor],
          labels: tf.Tensor,
          loss_weights: list[float] = None,
          mode='loss',
          *args, **kwargs) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:

    loss_dict = {}
    for loss in loss_list:
        if type(loss) == str:
            if re.match('[Aa][Dd][Ee]', loss):
                loss_dict['ADE({})'.format(mode)] = ADE(
                    model_outputs[0], labels)

            elif re.match('[Ff][Dd][Ee]', loss):
                loss_dict['FDE({})'.format(mode)] = FDE(
                    model_outputs[0], labels)

            elif re.match('[Dd][Ii][Ff]', loss):
                order = 2 if not 'diff_order' in kwargs.keys() \
                    else kwargs['diff_order']
                weights = [min(1.0, 5 * 10 ** -o) for o in range(order+1)] \
                    if not 'diff_weights' in kwargs.keys() \
                    else kwargs['diff_weights']
                loss_dict['Diff'] = tf.reduce_sum(
                    tf.stack(weights) *
                    tf.stack(diff(model_outputs[0], labels, order)))

        elif callable(loss):
            loss_dict[loss.__name__ + '({})'.format(mode)] = loss(model_outputs, labels,
                                                                  *args, **kwargs)

    if loss_weights is None:
        loss_weights = tf.ones(len(loss_dict))

    if len(loss_weights) != len(loss_dict):
        raise ValueError('Incorrect weights')

    summary = tf.matmul(tf.expand_dims(list(loss_dict.values()), 0),
                        tf.expand_dims(loss_weights, 1))
    summary = tf.reshape(summary, ())
    return summary, loss_dict


def ADE(pred: tf.Tensor, GT: tf.Tensor) -> tf.Tensor:
    """
    Calculate `ADE` or `minADE`.

    :param pred: pred traj, shape = `[batch, (K), pred, 2]`
    :param GT: ground truth future traj, shape = `[batch, pred, 2]`
    :return loss_ade:
        Return `ADE` when input_shape = [batch, pred_frames, 2];
        Return `minADE` when input_shape = [batch, K, pred_frames, 2].
    """

    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)

    if pred.ndim == 3:
        pred = pred[:, tf.newaxis, :, :]

    all_ade = tf.reduce_mean(
        tf.linalg.norm(
            pred - GT[:, tf.newaxis, :, :],
            ord=2, axis=-1
        ), axis=-1)
    best_ade = tf.reduce_min(all_ade, axis=1)
    return tf.reduce_mean(best_ade)


def FDE(pred, GT) -> tf.Tensor:
    """
    Calculate `FDE` or `minFDE`

    :param pred: pred traj, shape = `[batch, pred, 2]`
    :param GT: ground truth future traj, shape = `[batch, pred, 2]`
    :return fde:
        Return `FDE` when input_shape = [batch, pred_frames, 2];
        Return `minFDE` when input_shape = [batch, K, pred_frames, 2].
    """
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)

    t = pred.shape[-2]
    f = tf.gather(pred, [t-1], axis=-2)
    f_gt = tf.gather(GT, [t-1], axis=-2)
    return ADE(f, f_gt)


def context(pred, maps, paras, pred_bias=None) -> tf.Tensor:
    """
    Context loss.
    See details in "CSCNet: Contextual Semantic Consistency Network 
    for Trajectory Prediction in Crowded Spaces".

    :param pred: pred traj, shape = `[batch, pred, 2] or [batch, K, pred, 2]`
    :param maps: energy map, shape = `[batch, h, w]`
    :param paras: mapping function paras [[Wx, Wy], [bx, by]]
    :param pred_bias: bias for prediction, shape = `[batch, 2]`
    :return loss_context: context loss
    """

    if type(pred_bias) == type(None):
        pred_bias = tf.zeros([pred.shape[0], 2], dtype=tf.float32)
    if len(pred_bias.shape) == 2:
        pred_bias = tf.expand_dims(pred_bias, axis=1)

    if len(pred.shape) == 3:
        W = tf.reshape(paras[:, 0, :], [-1, 1, 2])
        b = tf.reshape(paras[:, 1, :], [-1, 1, 2])
    elif len(pred.shape) == 4:
        W = tf.expand_dims(tf.reshape(paras[:, 0, :], [-1, 1, 2]), axis=1)
        b = tf.expand_dims(tf.reshape(paras[:, 1, :], [-1, 1, 2]), axis=1)

    # from real positions to grid positions, shape = [batch, pred, 2]
    pred_grid = tf.cast((pred - b) * W, tf.int32)

    if len(pred.shape) == 3:
        center_grid = tf.cast((pred_bias - b) * W, tf.int32)
    elif len(pred.shape) == 4:
        center_grid = tf.cast(
            (tf.expand_dims(pred_bias, axis=1) - b) * W, tf.int32)
    final_grid = pred_grid - center_grid + \
        tf.cast(maps.shape[-1]/2, tf.int32)

    final_grid = tf.maximum(final_grid, tf.zeros_like(final_grid))
    final_grid = tf.minimum(
        final_grid, (maps.shape[-1]-1) * tf.ones_like(final_grid))

    if len(pred.shape) == 3:
        sel = tf.gather_nd(maps, final_grid, batch_dims=1)
    elif len(pred.shape) == 4:
        sel = tf.gather_nd(tf.repeat(tf.expand_dims(
            maps, axis=1), pred.shape[1], axis=1), final_grid, batch_dims=2)
    context_loss_mean = tf.reduce_mean(sel)
    return context_loss_mean


def diff(pred, GT, ordd=2) -> list[tf.Tensor]:
    """
    Diff loss. (It is unused now.)

    :param pred: pred traj, shape = `[(K,) batch, pred, 2]`
    :param GT: ground truth future traj, shape = `[batch, pred, 2]`
    :return loss: a list of Tensor, `len(loss) = ord + 1`
    """
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)

    pred_diff = difference(pred, ordd=ordd)
    GT_diff = difference(GT, ordd=ordd)

    loss = []
    for pred_, gt_ in zip(pred_diff, GT_diff):
        loss.append(ADE(pred_, gt_))

    return loss


def difference(trajs: tf.Tensor, direction='back', ordd=1) -> list[tf.Tensor]:
    """
    :param trajs: trajectories, shape = `[(K,) batch, pred, 2]`
    :param direction: string, canbe `'back'` or `'forward'`
    :param ord: repeat times

    :return result: results list, `len(results) = ord + 1`
    """
    outputs = [trajs]
    for repeat in range(ordd):
        outputs_current = \
            outputs[-1][:, :, 1:, :] - outputs[-1][:, :, :-1, :] if len(trajs.shape) == 4 else \
            outputs[-1][:, 1:, :] - outputs[-1][:, :-1, :]
        outputs.append(outputs_current)
    return outputs
    