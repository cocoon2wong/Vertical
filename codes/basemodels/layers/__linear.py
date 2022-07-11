"""
@Author: Conghao Wong
@Date: 2021-12-21 15:19:11
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:16:17
@Description: Linear trajectory prediction layers.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, obs_frames, pred_frames, diff=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.h = obs_frames
        self.f = pred_frames
        self.diff = diff

        if self.diff == 0:
            P = tf.linalg.diag(tf.ones(self.h))
        else:
            P = tf.linalg.diag(tf.nn.softmax(
                [(i+1)**self.diff for i in range(self.h)]))

        self.x = tf.range(self.h, dtype=tf.float32)
        self.x_p = tf.range(self.f, dtype=tf.float32) + self.h
        A = tf.transpose(tf.stack([
            tf.ones([self.h]),
            self.x
        ]))
        self.A_p = tf.transpose(tf.stack([
            tf.ones([self.f]),
            self.x_p
        ]))
        self.W = tf.linalg.inv(tf.transpose(A) @ P @ A) @ tf.transpose(A) @ P

    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Linear prediction

        :param inputs: input trajs, shape = (batch, obs, 2)
        :param results: linear pred, shape = (batch, pred, 2)
        """
        x = inputs[:, :, 0:1]
        y = inputs[:, :, 1:2]

        Bx = self.W @ x
        By = self.W @ y

        results = tf.stack([
            self.A_p @ Bx,
            self.A_p @ By,
        ])

        results = tf.transpose(results[:, :, :, 0], [1, 2, 0])
        return results[:, -self.f:, :]


class LinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        """
        Piecewise linear interpolation
        (Results do not contain the start point)
        """

        super().__init__(*args, **kwargs)

    def call(self, index, value):
        """
        Piecewise linear interpolation
        (Results do not contain the start point)

        :param index: index, shape = `(n)`, where `m = index[-1] - index[0]`
        :param value: values, shape = `(..., n, 2)`
        :return yp: linear interpolations, shape = `(..., m, 2)`
        """

        x = index
        y = value

        linear_results = []
        for output_index in range(x.shape[0] - 1):
            p_start = x[output_index]
            p_end = x[output_index+1]

            # shape = (..., 2)
            start = tf.gather(y, output_index, axis=-2)
            end = tf.gather(y, output_index+1, axis=-2)

            for p in tf.range(p_start+1, p_end+1):
                linear_results.append(tf.expand_dims(
                    (end - start) * (p - p_start) / (p_end - p_start)
                    + start, axis=-2))   # (..., 1, 2)

        # shape = (..., n, 2)
        return tf.concat(linear_results, axis=-2)
