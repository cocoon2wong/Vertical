"""
@Author: Conghao Wong
@Date: 2021-12-21 15:25:47
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:15:23
@Description: Encoding layers.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from .__transformLayers import _BaseTransformLayer


class TrajEncoding(tf.keras.layers.Layer):
    """
    Encode trajectories into the traj feature
    """

    def __init__(self, units: int = 64,
                 activation=None,
                 transform_layer: _BaseTransformLayer = None,
                 *args, **kwargs):
        """
        Init a trajectory encoding module

        :param units: feature dimension
        :param activation: activations used in the output layer
        :param transform_layer: controls if encode trajectories \
            with some transform methods (like FFTs)
        """

        super().__init__(*args, **kwargs)

        self.Tlayer = None

        if transform_layer:
            self.Tlayer = transform_layer
            self.fc2 = tf.keras.layers.Dense(units, tf.nn.relu)

        self.fc1 = tf.keras.layers.Dense(units, activation)

    def call(self, trajs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode trajectories into the high-dimension features

        :param trajs: trajs, shape = `(batch, N, 2)`
        :return features: features, shape = `(batch, N, units)`
        """
        if self.Tlayer:
            t = self.Tlayer(trajs)
            fc2 = self.fc2(t)
            return self.fc1(fc2)

        else:
            return self.fc1(trajs)


class ContextEncoding(tf.keras.layers.Layer):
    """
    Encode context maps into the context feature
    """

    def __init__(self, output_channels: int,
                 units: int = 64,
                 activation=None,
                 *args, **kwargs):
        """
        Init a context encoding module

        :param output_channels: output channels 
        :param units: output feature dimension
        :param activation: activations used in the output layer
        """

        super().__init__(*args, **kwargs)

        self.pool = tf.keras.layers.MaxPooling2D([5, 5])
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(output_channels * units, activation)
        self.reshape = tf.keras.layers.Reshape((output_channels, units))

    def call(self, context_map: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode context maps into context features

        :param context_map: maps, shape = `(batch, a, a)`
        :return feature: features, shape = `(batch, output_channels, units)`
        """
        pool = self.pool(context_map[:, :, :, tf.newaxis])
        flat = self.flatten(pool)
        fc = self.fc(flat)
        return self.reshape(fc)
