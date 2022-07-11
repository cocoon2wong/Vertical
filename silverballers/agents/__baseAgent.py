"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:55
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-22 19:55:50
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import codes as C
import tensorflow as tf

from ..__args import AgentArgs


class BaseAgentStructure(C.training.Structure):

    model_type = None

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = AgentArgs(terminal_args)
        self.important_args += ['Kc', 'key_points', 'depth', 'preprocess']

        self.set_inputs('obs')
        self.set_labels('pred')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)

        if self.args.metric == 'fde':
            self.set_metrics(self.min_FDE)
        elif self.args.metric == 'ade':
            self.set_metrics(self.l2_loss)
        else:
            raise ValueError(self.log('Metric error!', level='error'))

        self.set_metrics_weights(1.0)

    @property
    def p_index(self) -> tf.Tensor:
        """
        Time step of predicted key points.
        """
        p_index = [int(i) for i in self.args.key_points.split('_')]
        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        """
        Length of predicted key points.
        """
        return len(self.p_index)

    def set_model_type(self, new_type):
        self.model_type = new_type

    def create_model(self, *args, **kwargs):
        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               id_depth=self.args.depth,
                               keypoints_number=self.p_len,
                               structure=self)

    def l2_loss(self, outputs: list[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:
        """
        L2 distance between predictions and labels on predicted key points
        """
        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return C.training.loss.ADE(outputs[0], labels_pickled)

    def min_FDE(self, outputs: list[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:
        """
        minimum FDE among all predictions
        """
        # shape = (batch, Kc*K)
        distance = tf.linalg.norm(
            outputs[0][:, :, -1, :] -
            tf.expand_dims(labels[:, -1, :], 1), axis=-1)

        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
