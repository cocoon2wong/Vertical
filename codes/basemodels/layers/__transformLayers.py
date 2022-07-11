"""
@Author: Conghao Wong
@Date: 2022-06-22 15:47:41
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-22 16:22:16
@Description: Transformation layers.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from ..wavetf import WaveTFFactory


class _BaseTransformLayer(tf.keras.layers.Layer):
    """
    Calculate some Transform on (a batch of) trajectories.
    """

    def __init__(self, Oshape: tuple[int, int],
                 *args, **kwargs):
        """
        init

        :param Oshape: original shape of the layer inputs.\
            It dose not contain the `batch_size`
        """

        super().__init__(*args, **kwargs)

        self.steps = Oshape[0]
        self.channels = Oshape[1]

        # original and transformed shapes
        self.__Oshape = Oshape
        self.__Tshape = self.set_Tshape()

        # switch implementation mode
        # canbe `bigbatch` or `repeat` or `direct`
        self.mode = 0

        self.trainable = False

    @property
    def Oshape(self) -> tuple[int, int]:
        """
        Original shape of the input sequences.
        It does not contain the `batch_size` item.
        For example, `(steps, channels)`.
        """
        return (self.__Oshape[0], self.__Oshape[1])

    @property
    def Tshape(self) -> tuple[int, int]:
        """
        Shape after applying transfroms on the input sequences.
        It does not contain the `batch_size` item.
        For example, `(steps, channels)`.
        """
        return (self.__Tshape[0], self.__Tshape[1])

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        """
        Set output shape after the given transform on the input \
            trajectories, whose shapes are `(batch, steps, channels)`.
        It should be calculated with `self.steps` and `self.channels`.
        """
        raise NotImplementedError

    def call(self, inputs: tf.Tensor, *args, **kwargs):

        # calculate with a big batch size
        if self.mode == 0:
            shape_original = None
            if inputs.ndim == 4:
                shape_original = tf.shape(inputs)
                inputs = tf.reshape(inputs, [shape_original[0]*shape_original[1],
                                             shape_original[2],
                                             shape_original[3]])

            outputs = self.kernel_function(inputs, *args, **kwargs)

            if shape_original is not None:
                shape_new = tf.shape(outputs)
                outputs = tf.reshape(outputs, [shape_original[0],
                                               shape_original[1],
                                               shape_new[1],
                                               shape_new[2]])

        # calculate recurrently
        elif self.mode == 1:
            reshape = False
            if inputs.ndim == 3:
                reshape = True
                inputs = inputs[tf.newaxis, :, :, :]

            outputs = []
            for batch_inputs in inputs:
                outputs.append(self.kernel_function(
                    batch_inputs,
                    *args, **kwargs))

            if reshape:
                outputs = outputs[0]

            outputs = tf.cast(outputs, tf.float32)

        # calculate directly
        elif self.mode == 2:
            outputs = self.kernel_function(inputs, *args, **kwargs)

        else:
            raise ValueError('Mode does not exist.')

        return outputs

    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs):
        """
        Calculate any kinds of transform on a batch of trajectories.

        :param inputs: a batch of agents' trajectories, \
            shape is `(batch, steps, channels)`
        :return r: the transform of trajectories
        """
        raise NotImplementedError


class FFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:
        """
        Run FFT on a batch of trajectories.

        :param inputs: batch inputs, \
            shape = `(batch, steps, channels)`
        :return fft: fft results (real and imag), \
            shape = `(batch, steps, 2*channels)`
        """

        ffts = []
        for index in range(0, inputs.shape[-1]):
            seq = tf.cast(tf.gather(inputs, index, axis=-1), tf.complex64)
            seq_fft = tf.signal.fft(seq)
            ffts.append(tf.expand_dims(seq_fft, -1))

        ffts = tf.concat(ffts, axis=-1)
        return tf.concat([tf.math.real(ffts), tf.math.imag(ffts)], axis=-1)


class IFFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):

        real = tf.gather(inputs, tf.range(
            0, self.channels), axis=-1)
        imag = tf.gather(inputs, tf.range(
            self.channels, 2*self.channels), axis=-1)

        ffts = []
        for index in range(0, real.shape[-1]):
            r = tf.gather(real, index, axis=-1)
            i = tf.gather(imag, index, axis=-1)
            ffts.append(
                tf.expand_dims(
                    tf.math.real(
                        tf.signal.ifft(
                            tf.complex(r, i)
                        )
                    ), axis=-1
                )
            )

        return tf.concat(ffts, axis=-1)


class Haar1D(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        if self.Oshape[0] % 2 == 1:
            raise ValueError('`steps` in haar wavelet must be an even')

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):

        # (batch, steps, channels) -> (batch, steps//2, 2*channels)
        haar = self.haar.call(inputs)

        return haar


class InverseHaar1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        if self.Oshape[0] % 2 == 1:
            raise ValueError('`steps` in haar wavelet must be an even')

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:

        # (batch, steps//2, 2*channels) -> (batch, steps, channels)
        r = self.haar.call(inputs)

        return r


class DB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):
        return self.daub.call(inputs)


class InverseDB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):
        return self.daub.call(inputs)
        