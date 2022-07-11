"""
@Author: Conghao Wong
@Date: 2022-06-21 19:24:34
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:55:40
@Description: Methods to get input tensors from dataset-related structures.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .__agent import Agent


def get_inputs_by_type(input_agents: list[Agent],
                       type_name: str) -> tf.Tensor:
    """
    Get model inputs from a list of `Agent`-like objects.

    :param input_agents: a list of `Agent` objects or their subclass-objects
    :param type_name: inputs names, accept `'TRAJ'`, `'MAP'`, `'MAPPARA'`,
        `'DEST'`, and `'GT'`
    :return inputs: a tensor of stacked inputs
    """
    if type_name == 'TRAJ':
        call = _get_obs_traj
    elif type_name == 'MAP':
        call = _get_context_map
    elif type_name == 'MAPPARA':
        call = _get_context_map_paras
    elif type_name == 'DEST':
        call = _get_dest_traj
    elif type_name == 'GT':
        call = _get_gt_traj
    return call(input_agents)


def _get_obs_traj(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get observed trajectories from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of observed trajectories
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare trajectories...'):
        inputs.append(agent.traj)
    return tf.cast(inputs, tf.float32)


def _get_gt_traj(input_agents: list[Agent],
                 destination=False) -> tf.Tensor:
    """
    Get groundtruth trajectories from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of gt trajectories
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare groundtruth...'):
        if destination:
            inputs.append(np.expand_dims(agent.groundtruth[-1], 0))
        else:
            inputs.append(agent.groundtruth)

    return tf.cast(inputs, tf.float32)


def _get_dest_traj(input_agents: list[Agent]) -> tf.Tensor:
    return _get_gt_traj(input_agents, destination=True)


def _get_context_map(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get context map from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of maps
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.Map)
    return tf.cast(inputs, tf.float32)


def _get_context_map_paras(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get parameters of context map from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of map paras
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.real2grid)
    return tf.cast(inputs, tf.float32)
