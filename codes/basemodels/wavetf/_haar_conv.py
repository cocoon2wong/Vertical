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
        return t1 # out: (b, c, 2*nx)
    def haar_1(self, t1):
        ## t1: (b, c, x) with (x % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xb = 2.0 * t1[:,:,-1:]
        col1_b = col1_xb - t1[:,:,-2:-1] # 2*x_{n-1} - x_{n-2}
        s1 = tf.concat([t1, col1_b], axis=-1) # out: (b, c, 2*nx)
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
        ## s1: (b, c, 2*nx)
        s1 = tf.reshape(s1, [-1, 2*self.nx, 1]) # out: (b*c, 2*nx, 1)
        # build kernels and apply to rows
        k1l = tf.reshape(haar_ker[:,0], (2, 1, 1))
        k1h = tf.reshape(haar_ker[:,1], (2, 1, 1))
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b*c, nx, 2)
        r = tf.reshape(r, [self.bs, self.cn, self.nx, 2]) # out: (b, c, nx, 2)
        r = tf.transpose(r, [0, 2, 3, 1]) # out: (b, nx, 2, c)
        r = tf.reshape(r, [self.bs, self.nx, 2*self.cn]) # out: (b, nx, 2*c)
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
        # input: (b, nx, 2*c)
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        t1 = tf.reshape(input, [self.bs, self.ox, self.cn])
        # out: (b, ox, c)
        t1 = tf.transpose(t1, perm=[0, 2, 1]) # out: (b, c, ox)
        t1 = tf.reshape(t1, [-1, self.ox, 1]) # out: (b*c, ox, 1)
        # apply kernel to rows
        k1l = tf.reshape(haar_ker[:,0], (2, 1, 1))
        k1h = tf.reshape(haar_ker[:,1], (2, 1, 1))
        rl = tf.nn.conv1d(t1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(t1, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b*c, qx, 4)
        r = tf.reshape(r, [self.bs, self.cn, self.ox]) # out: (b, c, ox)
        r = tf.transpose(r, [0, 2, 1]) # out: (b, ox, c)
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
    def haar_0(self, t1):
        ## t1: (..., z) with (z % 2) == 0
        return t1 # out: (..., 2*nz)
    def haar_1(self, t1):
        ## t1: (..., z) with (z % 2) == 1
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xb = 2.0 * t1[... , -1:]
        col1_b = col1_xb - t1[... , -2:-1] # 2*x_{n-1} - x_{n-2}
        s1 = tf.concat([t1, col1_b], axis=-1) # out: (..., 2*nz)
        return s1
    def kernel_function(self, input):
        # input: (b, x, y, c)
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        mod_x = self.ox % 2
        mod_y = self.oy % 2
        ## pass 1: transform rows
        t1 = tf.transpose(input, perm=[0, 3, 1, 2]) # out: (b, c, ox, oy)
        if (mod_y == 0) :
            s1 = self.haar_0(t1)
        else :
            s1 = self.haar_1(t1)
        ## s1: (b, c, ox, 2*ny)
        s1 = tf.reshape(s1, [-1, 2*self.ny, 1])
        ## s1: (b*c*ox, 2*ny, 1)
        # build kernels and apply to rows
        k1l = tf.reshape(haar_ker[:,0], (2, 1, 1))
        k1h = tf.reshape(haar_ker[:,1], (2, 1, 1))
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        s1 = tf.concat((rl, rh), axis=-1) # out: (b*c*ox, ny, 2)
        s1 = tf.reshape(s1, [self.bs, self.cn, self.ox, self.ny, 2])
        # out: (b, c, ox, ny, 2_y)
        ## transform columns
        t2 = tf.transpose(s1, perm=[0, 1, 3, 4, 2]) # out: (b, c, ny, 2_y, ox)
        if (mod_x == 0) :
            s2 = self.haar_0(t2)
        else :
            s2 = self.haar_1(t2)
        ## s2: (b, c, ny, 2_y, 2*nx)
        s2 = tf.reshape(s2, [-1, 2*self.nx, 1])
        # out: (b*c*ny*2_y, 2*nx, 1)
        # build kernels and apply kernel to columns
        rl = tf.nn.conv1d(s2, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s2, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b*c*ny*2_y, nx, 2_x)
        r = tf.reshape(r, [self.bs, self.cn, self.ny, 2, self.nx, 2])
        # out: (b, c, ny, 2_y, nx, 2_x)
        r = tf.transpose(r, perm=[0, 4, 2, 3, 5, 1]) # out: (b, nx, ny, 2_y, 2_x, c)
        r = tf.reshape(r, [self.bs, self.nx, self.ny, 4*self.cn])
        # out: (b, ny, nx, 4*c)
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
        # input: (b, nx, ny, 4*c)
        # choose float precision accordingly to input
        haar_ker = tf.cast(self.haar_ker, tf.float32) if (input.dtype==tf.float32) else self.haar_ker
        t1 = tf.reshape(input, [self.bs, self.nx, self.ny, 2, 2, self.cn])
        # out: (b, x, y, 2_y, 2_x, c)
        t1 = tf.transpose(t1, perm=[0, 5, 2, 3, 1, 4])
        # out: (b, c, y, 2_y, x, 2_x)
        t1 = tf.reshape(t1, [-1, self.ox, 1])
        # out: (b*c*oy, ox, 1)
        # apply kernel to x
        k1l = tf.reshape(haar_ker[:,0], (2, 1, 1))
        k1h = tf.reshape(haar_ker[:,1], (2, 1, 1))
        rl = tf.nn.conv1d(t1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(t1, k1h, stride=2, padding='VALID')
        s1 = tf.concat((rl, rh), axis=-1) # out: (b*c*oy, qx, 4)
        s1 = tf.reshape(s1, [self.bs, self.cn, self.oy, self.ox])
        # out: (b, c, oy, ox)
        s1 = tf.transpose(s1, perm=[0, 1, 3, 2]) # out: (b, c, ox, oy)
        s1 = tf.reshape(s1, [-1, self.oy, 1])
        # out: (b*c*ox, oy, 1)
        # apply kernel to y
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b*c*ox, qy, 4)
        r = tf.reshape(r, [self.bs, self.cn, self.ox, self.oy])
        # out: (b, c, ox, oy)
        r = tf.transpose(r, [0, 2, 3, 1]) # out: (b, ox, oy, c)
        return r

