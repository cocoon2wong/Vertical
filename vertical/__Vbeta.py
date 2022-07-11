"""
@Author: Conghao Wong
@Date: 2022-06-23 10:23:53
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 11:24:46
@Description: Second stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers, transformer
from silverballers.handlers import BaseHandlerModel, BaseHandlerStructure

from .__args import VArgs


class VBModel(BaseHandlerModel):
    """
    Spectrum Interpolation Sub-network
    ---

    The second stage V^2-Net sub-network.
    It is used to interpolate agents' entire predictions
    by considering their interactions details.
    It also implements on agents' spectrums instead of
    their trajectories.
    """

    def __init__(self, Args: VArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler, key_points,
                         structure, *args, **kwargs)

        # Layers
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.linear_interpolation = layers.LinearInterpolation()

        self.fft = layers.FFTLayer((self.args.obs_frames, 2))

        self.te = layers.TrajEncoding(units=64,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.fft)

        self.ce = layers.ContextEncoding(units=64,
                                         output_channels=self.args.obs_frames,
                                         activation=tf.nn.tanh)

        self.transformer = transformer.Transformer(num_layers=4,
                                                   d_model=128,
                                                   num_heads=8,
                                                   dff=512,
                                                   input_vocab_size=None,
                                                   target_vocab_size=4,
                                                   pe_input=Args.obs_frames,
                                                   pe_target=Args.obs_frames + Args.pred_frames,
                                                   include_top=True)

        self.decoder = layers.IFFTLayer(
            (self.args.obs_frames + self.args.pred_frames, 2))

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])

        # transformer target shape = (batch, obs+pred, 4)
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # add the last obs point to finish linear interpolation
        linear_pred = self.linear_interpolation.call(
            keypoints_index, keypoints)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        t_outputs = self.fft.call(traj)

        # transformer output shape = (batch, obs+pred, 4)
        p_fft, _ = self.transformer.call(t_inputs,
                                         t_outputs,
                                         training=training)

        # decode
        p = self.decoder.call(p_fft)

        return p[:, self.args.obs_frames:, :]


class VB(BaseHandlerStructure):
    """
    Training structure for the second stage sub-network
    """

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.set_model_type(new_type=VBModel)
        self.args = VArgs(terminal_args)
