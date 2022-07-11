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
from ._base_wavelets import DirWaveLayer1D, InvWaveLayer1D, DirWaveLayer2D, InvWaveLayer2D

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
## input: (b, x, c) --> output: (b, nx, 2*c)

class HaarWaveLayer1D(DirWaveLayer1D):
    """1D direct Haar trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Haar kernel
        s2 = math.sqrt(2) * .5 # 1/sqrt(2)
        self.haar_ker = tf.constant([s2, s2, s2, -s2], shape=(2,2), dtype=tf.float64)
        ## call constructor
        super(DirWaveLayer1D, self).__init__(**kwargs)
    ########################################################################
    ## Haar wavelet
    ########################################################################
    def haar_0(self, t1):
        ## t1: (b, c, x) with (x % 2) == 0
        return tf.reshape(t1, [-1, self.cn, self.nx, 2]) # out: (b, c, nx, 2)
    def haar_1(self, t1):
        ## t1: (b, c, x) with (x % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xb = 2.0 * t1[:,:,-1:]
        col1_b = col1_xb - t1[:,:,-2:-1] # 2*x_{n-1} - x_{n-2}
        s1 = tf.concat([t1, col1_b], axis=-1)
        # group by 2
        s1 = tf.reshape(s1, [-1, self.cn, self.nx, 2]) # out: (b, c, nx, 2)
        return s1
    def kernel_function(self, input):
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        mod_x = self.ox % 2
        # input: (b, x, c)
        t1 = tf.transpose(input, perm=[0, 2, 1]) # out: (b, c, x)
        ## prepare data
        if (mod_x == 0) :
            s1 = self.haar_0(t1)
        else :
            s1 = self.haar_1(t1)
        ## s1: (b, c, nx, 2)
        # apply kernel to rows
        r = s1 @ haar_ker # out: (b, c, nx, 2)
        r = tf.transpose(r, perm=[0, 2, 3, 1]) # out: (b, nx, 2, c)
        r = tf.reshape(r, [self.bs, self.nx, 2*self.cn])
        return r

########################################################################
# Inverse wavelet transforms
## input: (b, x, 2*c) --> output: (b, 2*x, c)
## (with input[b,x,:c] being the channels for L wavelet)


class InvHaarWaveLayer1D(InvWaveLayer1D):
    """1D inverse Haar trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Haar kernel
        s2 = math.sqrt(2) * .5 # 1/sqrt(2)
        self.haar_ker = tf.constant([s2, s2, s2, -s2], shape=(2,2), dtype=tf.float64)
        ## call constructor
        super(InvWaveLayer1D, self).__init__(**kwargs)
    ########################################################################
    ## Haar wavelet
    ########################################################################
    def kernel_function(self, input):
        # input: (b, x, 2*c)
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        t1 = tf.reshape(input, [self.bs, self.nx, 2, self.cn])
        # out: (b, x, 2, c)
        t1 = tf.transpose(t1, perm=[0, 3, 1, 2])
        # out: (b, c, x, 2)
        # apply kernel to rows
        r = t1 @ haar_ker # out: (b, c, x, 2)
        r = tf.reshape(r, [self.bs, self.cn, self.ox]) # out: (b, c, 2*x)
        r = tf.transpose(r, perm=[0, 2, 1])
        # out: (b, 2*x, c)
        return r

    
########################################################################
# 2D wavelet
########################################################################

########################################################################
# Direct wavelet transform
########################################################################
## apply wavelet transform to each channel of the input tensor
## input: (b, x, y, c) --> output: (b, nx, ny, 4*c)
## (with outpuy[b,x,y,:c] being the channels for LL wavelet)
    
class HaarWaveLayer2D(DirWaveLayer2D):
    """2D direct Haar trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Haar kernel
        s2 = math.sqrt(2) * .5 # 1/sqrt(2)
        self.haar_ker = tf.constant([s2, s2, s2, -s2], shape=(2,2), dtype=tf.float64)
        ## call constructor
        super(DirWaveLayer2D, self).__init__(**kwargs)
    ########################################################################
    ## Haar wavelet
    ########################################################################
    def haar1_0(self, t1):
        ## t1: (b, c, x, y) with (y % 2) == 0
        return tf.reshape(t1, [self.bs, self.cn, self.ox, self.ny, 2]) # out: (b, c, x, ny, 2)
    def haar1_1(self, t1):
        ## t1: (b, c, x, y) with (y % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xb = 2.0 * t1[:,:,:,-1:]
        col1_b = col1_xb - t1[:,:,:,-2:-1] # 2*x_{n-1} - x_{n-2}
        s1 = tf.concat([t1, col1_b], axis=-1)
        # group by 2
        s1 = tf.reshape(s1, [self.bs, self.cn, self.ox, self.ny, 2]) # out: (b, c, x, ny, 2)
        return s1
    def haar2_0(self, t2): 
        ## t2: (b, 2, c, ny, x) with (y % 2) == 0
        return tf.reshape(t2, [self.bs, 2, self.cn, self.ny, self.nx, 2]) # out: (b, 2, c, ny, nx, 2)
    def haar2_1(self, t2): 
        ## t2: (b, 2, c, ny, x) with (y % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab 
        col2_xb = 2.0 * t2[:,:,:,:,-1:]
        col2_b = col2_xb - t2[:,:,:,:,-2:-1] # 2*x_{n-1} - x_{n-2}
        s2 = tf.concat([t2, col2_b], axis=-1)
        # group by 2
        s2 = tf.reshape(s2, [self.bs, 2, self.cn, self.ny, self.nx, 2]) # out: (b, 2, c, ny, nx, 2)
        return s2
    def kernel_function(self, input):
        # input: (b, x, y, c)
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        mod_x = self.ox % 2
        mod_y = self.oy % 2
        ## pass 1: transform rows
        t1 = tf.transpose(input, perm=[0, 3, 1, 2]) # out: (b, c, x, y)
        if (mod_y == 0) :
            s1 = self.haar1_0(t1)
        else :
            s1 = self.haar1_1(t1)
        ## s1: (b, c, x, ny, 2)
        # apply kernel to rows
        s1 = s1 @ haar_ker # out: (b, c, x, ny, 2)
        ## pass 2: transform columns
        t2 = tf.transpose(s1, perm=[0, 4, 1, 3, 2]) # out: (b, 2, c, ny, x)
        if (mod_x == 0) :
            s2 = self.haar2_0(t2)
        else :
            s2 = self.haar2_1(t2)
        ## s2: (b, 2, c, ny, nx, 2)
        # apply kernel to columns
        r = s2 @ haar_ker # out: (b, 2, c, ny, nx, 2)
        r = tf.transpose(r, perm=[0, 4, 3, 1, 5, 2]) # out: (b, nx, ny, 2_y, 2_x, c)
        r = tf.reshape(r, [self.bs, self.nx, self.ny, 4*self.cn])
        return r

########################################################################
# Inverse wavelet transform
########################################################################
## input: (b, x, y, 4*c) --> output: (b, 2*x, 2*y, c)
## (with input[b,x,y,:c] being the channels for LL wavelet)

class InvHaarWaveLayer2D(InvWaveLayer2D):
    """2D inverse Haar trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Haar kernel
        s2 = math.sqrt(2) * .5 # 1/sqrt(2)
        self.haar_ker = tf.constant([s2, s2, s2, -s2], shape=(2,2), dtype=tf.float64)
        ## call constructor
        super(InvWaveLayer2D, self).__init__(**kwargs)
    ########################################################################
    ## Inverse Haar wavelet
    ########################################################################
    def kernel_function(self, input):
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        # input: (b, x, y, 4*c)
        t1 = tf.reshape(input, [self.bs, self.nx, self.ny, 2, 2, self.cn])
        # out: (b, x, y, 2_y, 2_x, c)
        # apply kernel to x
        t1 = tf.transpose(t1, perm=[0, 5, 2, 3, 1, 4])
        # out: (b, c, y, 2_y, x, 2_x)
        s1 = t1 @ haar_ker # out: (b, c, y, 2, x, 2)
        # apply kernel to y
        s1 = tf.transpose(s1, perm=[0, 1, 4, 5, 2, 3])
        # out: (b, c, x, 2, y, 2)
        s1 = s1 @ haar_ker # out: (b, c, x, 2, y, 2)
        r = tf.reshape(s1, [self.bs, self.cn, self.ox, self.oy])
        # out: (b, c, 2*x, 2*y)
        r = tf.transpose(r, perm=[0, 2, 3, 1])
        # out: (b, 2*x, 2*y, c)
        return r

