# Copyright 2020 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
from tensorflow import keras

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
## input: (b, x, c) --> output: (b, nx, 2*c)


class DirWaveLayer1D(keras.layers.Layer):
    """Abstract class with general methods for 1D wavelet transforms"""
    # in : (b, x, c) --> out: (b, nx, 2*c)
    def call(self, batch):
        """Call the direct 1D wavelet

        :param batch: tensor of shape (batch_size, dim_x, chans)
        :returns: tensor of shape (batch_size, ceil(dim_x/2), 2*chans)
        :rtype: tensor

        """
        self.bs, self.ox, self.cn = batch.shape.as_list()
        if (self.bs is None) : self.bs = -1
        self.nx = math.ceil(self.ox / 2)
        self.qx = math.ceil(self.nx / 2)
        return self.kernel_function(batch)
    def compute_output_shape(self, input_shape):
        x = input_shape[1]
        cn = input_shape[2]
        h1 = math.ceil(x / 2)
        out_shape = tf.TensorShape([input_shape[0], h1, 2*cn])
        return(out_shape)

########################################################################
# Inverse wavelet transforms
## input: (b, x, 2*c) --> output: (b, 2*x, c)
## (with input[b,x,:c] being the channels for L wavelet)

class InvWaveLayer1D(keras.layers.Layer):
    """Abstract class with general methods for 1D inverse wavelet transforms"""
    # in : (b, x, 2*c) --> out: (b, 2x, c)
    def call(self, batch):
        """Call the inverse 1D wavelet

        :param batch: tensor of shape (batch_size, dim_x, 2*chans)
        :returns: tensor of shape (batch_size, 2*dim_x, chans)
        :rtype: tensor

        """
        self.bs, self.nx, self.cn = batch.shape.as_list()
        if (self.bs is None) : self.bs = -1
        self.ox = self.nx * 2
        self.cn = self.cn // 2
        return self.kernel_function(batch)
    def compute_output_shape(self, input_shape):
        x = input_shape[1]
        cn = input_shape[2]//2
        out_shape = tf.TensorShape([input_shape[0], 2*x, cn])
        return(out_shape)
    
    

########################################################################
# 2D wavelet
########################################################################

########################################################################
# Direct wavelet transform
########################################################################
## apply wavelet transform to each channel of the input tensor
## input: (b, x, y, c) --> output: (b, nx, ny, 4*c)
## (with output[b,x,y,:c] being the channels for LL wavelet)


class DirWaveLayer2D(keras.layers.Layer):
    """Abstract class with general methods for 2D wavelet transforms"""
    # in : (b, x, y, c) --> out: (b, nx, ny, 4*c)
    def call(self, batch):
        """Call the direct 2D wavelet.

        :param batch: tensor of shape (batch_size, dim_x, dim_y, chans)
        :returns: tensor of shape
            (batch_size, ceil(dim_x/2), ceil(dim_y/2), 4*chans),
            with output[:, :, :, :chans] being the LL channels
        :rtype: tensor

        """
        self.bs, self.ox, self.oy, self.cn = batch.shape.as_list()
        if (self.bs is None) : self.bs = -1
        self.nx, self.ny = map(lambda x: math.ceil(x / 2), [self.ox, self.oy])
        self.qx, self.qy = map(lambda x: math.ceil(x / 2), [self.nx, self.ny])
        return self.kernel_function(batch)
    def compute_output_shape(self, input_shape):
        bs = input_shape[0]
        x = input_shape[1]
        y = input_shape[2]
        cn = input_shape[3]
        h1 = math.ceil(x / 2)
        h2 = math.ceil(y / 2)
        out_shape = tf.TensorShape([bs, h1, h2, 4*cn])
        return(out_shape)
    
        
########################################################################
# Inverse wavelet transform
########################################################################
## input: (b, x, y, 4*c) --> output: (b, 2*x, 2*y, c)
## (with input[b,x,y,:c] being the channels for LL wavelet)

class InvWaveLayer2D(keras.layers.Layer):
    """Abstract class with general methods for 2D inverse wavelet transforms"""
    # in : (b, x, y, 4*c) --> out: (b, 2*x, 2*y, c)
    def call(self, batch):
        """Call the inverse 2D wavelet

        :param batch: tensor of shape
            (batch_size, dim_x, dim_y, 4*chans)
        :returns: tensor of shape
            (batch_size, 2*dim_x, 2*dim_y, chans)
        :rtype: tensor

        """
        self.bs, self.nx, self.ny, self.cn = batch.shape.as_list()
        if (self.bs is None) : self.bs = -1
        self.cn = self.cn // 4
        self.ox = self.nx * 2
        self.oy = self.ny * 2
        return self.kernel_function(batch)
    def compute_output_shape(self, input_shape):
        bs = input_shape[0]
        nx = input_shape[1]
        ny = input_shape[2]
        cn = input_shape[3] // 4
        out_shape = tf.TensorShape([bs, 2*nx, 2*ny, cn])
        return(out_shape)

