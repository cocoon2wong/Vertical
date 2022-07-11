"""
@Author: Conghao Wong
@Date: 2022-07-05 16:00:26
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 20:03:33
@Description: First stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model, layers, transformer
from silverballers.agents import BaseAgentStructure

from .__args import VArgs


class VAModel(Model):
    """
    Keypoints Estimation Sub-network
    ---

    The first stage V^2-Net sub-network.
    It is used to model agents' global plannings by considering
    agents' observed trajectory spectrums.
    The model takes agents' observed trajectories as the input,
    and output several keypoint trajectory spectrums finally.
    FFTs are applied before and after the model implementing.
    """

    def __init__(self, Args: VArgs,
                 feature_dim: int = 128,
                 id_depth: int = 64,
                 keypoints_number: int = 3,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args = Args

        # Parameters
        self.d = feature_dim
        self.n_key = keypoints_number
        self.d_id = id_depth    # depth of the random noise vector

        # Preprocess
        preprocess_list = ()
        for index, operation in enumerate(['Move', 'Scale', 'Rotate']):
            if self.args.preprocess[index] == '1':
                preprocess_list += (operation,)

        self.set_preprocess(*preprocess_list)
        self.set_preprocess_parameters(move=0)

        # Layers
        self.fft = layers.FFTLayer((self.args.obs_frames, 2))
        self.ifft = layers.IFFTLayer((self.n_key, 2))

        self.te = layers.TrajEncoding(units=self.d//2,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.fft)

        self.ie = layers.TrajEncoding(units=self.d//2,
                                      activation=tf.nn.tanh)

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(num_layers=4,
                                         d_model=self.d,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames,
                                         include_top=False)

        # Trainable adj matrix and gcn layer
        # See our previous work "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.adj_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Decoder layers
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(4 * self.n_key)
        self.decoder_reshape = tf.keras.layers.Reshape(
            [self.args.Kc, self.n_key, 4])

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs = inputs[0]   # (batch, obs, 2)
        bs = trajs.shape[0]

        # feature embedding and encoding -> (batch, obs, d)
        spec_features = self.te.call(trajs)

        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K
        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, obs, d)
            ids = tf.random.normal([bs, self.args.obs_frames, self.d_id])
            id_features = self.ie.call(ids)

            # transformer inputs
            t_inputs = self.concat([spec_features, id_features])
            t_outputs = self.fft.call(trajs)

            # transformer -> (batch, obs, d)
            behavior_features, _ = self.T.call(inputs=t_inputs,
                                               targets=t_outputs,
                                               training=training)

            # features -> (batch, Kc, d)
            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            m_features = self.gcn.call(behavior_features, adj)

            # predicted keypoints -> (batch, Kc, key, 2)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = self.decoder_reshape(y)

            y = self.ifft.call(y)
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=1)


class VA(BaseAgentStructure):
    """
    Training structure for the first stage sub-network
    """

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = VArgs(terminal_args)
        self.important_args = self.important_args[:-4]
        self.important_args += ['Kc']

        self.set_inputs('traj')
        self.set_labels('gt')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

        self.set_model_type(new_type=VAModel)
