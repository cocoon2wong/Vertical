"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 14:32:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import codes as C
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..__args import HandlerArgs


class BaseHandlerModel(C.basemodels.Model):

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Parameters
        self.asHandler = asHandler
        self.d = feature_dim
        self.points = points
        self.key_points = key_points

        if self.asHandler or key_points != 'null':
            pi = [int(i) for i in key_points.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

    def call_as_handler(self, inputs: list[tf.Tensor],
                        keypoints: tf.Tensor,
                        keypoints_index: tf.Tensor,
                        training=None, mask=None):
        """
        Call as the second stage handler model.
        Do NOT call this method when training.

        :param inputs: a list of trajs and context maps
        :param keypoints: predicted keypoints, shape is `(batch, K, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        """

        p_all = []
        K = keypoints.shape[1]

        for k in tqdm(range(K)):
            # single shape is (batch, pred, 2)
            p_all.append(self.call(inputs=inputs,
                                   keypoints=keypoints[:, k, :, :],
                                   keypoints_index=keypoints_index))

        return tf.transpose(tf.stack(p_all), [1, 0, 2, 3])

    def forward(self, model_inputs: list[tf.Tensor],
                training=None,
                *args, **kwargs):

        model_inputs_processed = self.pre_process(model_inputs, training)
        destination_processed = self.pre_process([model_inputs[-1]],
                                                 training,
                                                 use_new_para_dict=False)

        # only when training the single model
        if not self.asHandler:
            gt_processed = destination_processed[0]

            if self.key_points == 'null':
                index = np.random.choice(np.arange(self.args.pred_frames-1),
                                         self.points-1)
                index = tf.concat([np.sort(index),
                                   [self.args.pred_frames-1]], axis=0)
            else:
                index = tf.cast(self.points_index, tf.int32)

            points = tf.gather(gt_processed, index, axis=1)
            index = tf.cast(index, tf.float32)

            outputs = self.call(model_inputs_processed,
                                keypoints=points,
                                keypoints_index=index,
                                training=True)

        # use as the second stage model
        else:
            outputs = self.call_as_handler(model_inputs_processed,
                                           keypoints=destination_processed[0],
                                           keypoints_index=self.points_index,
                                           training=None)

        if not type(outputs) in [list, tuple]:
            outputs = [outputs]

        return self.post_process(outputs, training, model_inputs=model_inputs)


class BaseHandlerStructure(C.training.Structure):

    model_type = None

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = HandlerArgs(terminal_args)

        self.important_args += ['points']

        self.set_inputs('trajs', 'maps', 'paras', 'gt')
        self.set_labels('gt')

        self.set_loss('ade', 'diff')
        self.set_loss_weights(0.8, 0.2)

        if self.args.key_points == 'null':
            self.set_metrics('ade', 'fde')
            self.set_metrics_weights(1.0, 0.0)

        else:
            key_points = self.args.key_points
            pi = [int(i) for i in key_points.split('_')]
            self.keypoints_index = tf.cast(pi, tf.int32)

            self.set_metrics('ade', 'fde', self.l2_keypoints)
            self.set_metrics_weights(1.0, 0.0, 0.0)

    def set_model_type(self, new_type: type[BaseHandlerModel]):
        self.model_type = new_type

    def create_model(self, *args, **kwargs):
        return self.model_type(self.args, 128,
                               points=self.args.points,
                               key_points=self.args.key_points,
                               structure=self,
                               *args, **kwargs)

    def l2_keypoints(self, outputs: list[tf.Tensor],
                     labels: tf.Tensor,
                     *args, **kwargs) -> tf.Tensor:
        """
        l2 loss on the keypoints
        """
        # shapes of pickled tensors are (batch, n_key, 2)
        labels_pickled = tf.gather(labels, self.keypoints_index, axis=1)
        pred_pickled = tf.gather(outputs[0], self.keypoints_index, axis=1)

        return C.training.loss.ADE(pred_pickled, labels_pickled)
