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

class DaubWaveLayer1D(DirWaveLayer1D):
    """1D direct Daubechies-N=2 trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Daubechies kernel
        d = math.sqrt(2) * .125 # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3); h1 = d * (3+r3); h2 = d * (3-r3); h3 = d * (1-r3);
        g0 = h3; g1 = -h2; g2 = h1; g3 = -h0
        self.daubechies_ker = tf.constant([h0, g0, h1, g1, h2, g2, h3, g3], shape=(4,2), dtype=tf.float64)
        ## call constructor
        super(DirWaveLayer1D, self).__init__(**kwargs)
    ########################################################################
    ## Daubechies wavelet
    ########################################################################
    def daub_cols1(self, t1) :
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xa = 2.0 * t1[:,:,0:1]
        col1_xb = 2.0 * t1[:,:,-1:]
        col1_a = col1_xa - t1[:,:,1:2] # 2*x_0 - x_1
        col1_b = col1_xb - t1[:,:,-2:-1] # 2*x_{n-1} - x_{n-2}
        col1_c = col1_xb - t1[:,:,-3:-2] # 2*x_{n-1} - x_{n-3}
        return [col1_a, col1_b, col1_c]
    def daub_0(self, t1) :
        ## t1: (b, c, x) with (x % 2) == 0
        col1_a, col1_b, _ = self.daub_cols1(t1)
        s1 = tf.concat([col1_a, t1, col1_b], axis=-1) # out: (b, c, 2*nx)
        return s1
    def daub_1(self, t1) :
        ## t1: (b, c,x) with (x % 2) == 1
        col1_a, col1_b, col1_c = self.daub_cols1(t1)
        s1 = tf.concat([col1_a, t1, col1_b, col1_c], axis=-1)
        return s1
    def kernel_function(self, input):
        # choose float precision accordingly to input
        daubechies_ker = tf.cast(self.daubechies_ker, tf.float32) if (input.dtype==tf.float32) else self.daubechies_ker
        mod_x = self.ox % 4
        # input: (b, x, c)
        t1 = tf.transpose(input, perm=[0, 2, 1]) # out: (b, c, x)
        ## prepare data
        if (mod_x == 0) :
            s1 = self.daub_0(t1)
        else:
            s1 = self.daub_1(t1)
        ## s1: (b, c, 2*nx)
        nx_dim = s1.shape[2]
        s1 = tf.reshape(s1, [-1, nx_dim, 1]) # out: (b*c, 2*nx', 1)
        # build kernels and apply to rows
        k1l = tf.reshape(daubechies_ker[:,0], (4, 1, 1))
        k1h = tf.reshape(daubechies_ker[:,1], (4, 1, 1))
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b*c, nx, 2)
        r = tf.reshape(r, [self.bs, self.cn, self.nx, 2]) # out: (b, c, nx, 2)
        r = tf.transpose(r, [0, 2, 3, 1]) # out: (b, nx, 2, c)
        r = tf.reshape(r, [self.bs, self.nx, 2*self.cn]) # out: (b, nx, 2*c)
        return r

########################################################################
# Inverse wavelet transforms
## input: (b, x) --> output: (b, x), x must be even

class InvDaubWaveLayer1D(InvWaveLayer1D):
    """1D inverse Daubechies-N=2 trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Daubechies kernel
        d = math.sqrt(2) * .125 # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3); h1 = d * (3+r3); h2 = d * (3-r3); h3 = d * (1-r3);
        g0 = h3; g1 = -h2; g2 = h1; g3 = -h0
        self.daubechies_ker = tf.constant([h2, h3, g2, g3, h0, h1, g0, g1], shape=(4,2), dtype=tf.float64)
        # matrix for border effect 0 (begin)
        ker_0 = tf.constant([h0, h1, h2, g0, g1, g2, 0, 0, h0, 0, 0, g0], shape=(4,3), dtype=tf.float64)
        pad_0 = tf.constant([2., -1., 1., 0., 0., 1.], shape=(3,2), dtype=tf.float64)
        self.inv_bor_0 = tf.transpose(tf.linalg.pinv(ker_0 @ pad_0), [1, 0])
        # matrix for border effect 1 (end)
        ker_1 = tf.constant([h3, 0, 0, g3, 0, 0, h1, h2, h3, g1, g2, g3], shape=(4,3), dtype=tf.float64)
        pad_1 = tf.constant([1., 0., 0., 1., -1., 2.], shape=(3,2), dtype=tf.float64)
        self.inv_bor_1 = tf.transpose(tf.linalg.pinv(ker_1 @ pad_1), [1, 0])
        ## call constructor
        super(InvWaveLayer1D, self).__init__(**kwargs)
    def kernel_function(self, input):        
        # input: (b, nx, 2*c)
        # choose float precision accordingly to input
        daubechies_ker = tf.cast(self.daubechies_ker, tf.float32) if (input.dtype==tf.float32) else self.daubechies_ker
        inv_bor_0 = tf.cast(self.inv_bor_0, tf.float32) if (input.dtype==tf.float32) else self.inv_bor_0
        inv_bor_1 = tf.cast(self.inv_bor_1, tf.float32) if (input.dtype==tf.float32) else self.inv_bor_1
        #######################################
        ## reshape
        #######################################
        t1 = tf.reshape(input, [self.bs, self.ox, self.cn])
        # out: (b, ox, c)
        t1 = tf.transpose(t1, perm=[0, 2, 1]) # out: (b, c, ox)
        #######################################
        ## compute borders
        #######################################
        # border 0
        b_0 = t1[:,:,:4]
        r2_0 = b_0 @ inv_bor_0
        # border 1
        b_1 = t1[:,:,-4:]
        r2_1 = b_1 @ inv_bor_1
        #######################################
        ## transform core
        #######################################
        t1 = tf.reshape(t1, [-1, self.ox, 1]) # out: (b*c, ox, 1)
        # apply kernel to rows
        k1l = tf.reshape(daubechies_ker[:,0], (4, 1, 1))
        k1h = tf.reshape(daubechies_ker[:,1], (4, 1, 1))
        rl = tf.nn.conv1d(t1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(t1, k1h, stride=2, padding='VALID')
        r1 = tf.concat((rl, rh), axis=-1) # out: (b*c, qx, 4)
        r1 = tf.reshape(r1, [self.bs, self.cn, self.ox-2]) # out: (b, c, ox-2)
        #######################################
        ## merge core and borders
        #######################################
        r = tf.concat((r2_0, r1[:,:,1:-1], r2_1), axis=-1)
        # out: (b, c, nx)
        r = tf.transpose(r, perm=[0, 2, 1])
        # out: (b, nx, c)
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

class DaubWaveLayer2D(DirWaveLayer2D):
    """2D direct Daubechies-N=2 trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Daubechies kernel
        d = math.sqrt(2) * .125 # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3); h1 = d * (3+r3); h2 = d * (3-r3); h3 = d * (1-r3);
        g0 = h3; g1 = -h2; g2 = h1; g3 = -h0
        self.daubechies_ker = tf.constant([h0, g0, h1, g1, h2, g2, h3, g3], shape=(4,2), dtype=tf.float64)
        ## call constructor
        super(DirWaveLayer2D, self).__init__(**kwargs)
    ########################################################################
    ## Daubechies wavelet
    ########################################################################
    def daub_cols(self, t1) :
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        col1_xa = 2.0 * t1[... , 0:1]
        col1_xb = 2.0 * t1[... , -1:]
        col1_a = col1_xa - t1[... , 1:2] # 2*x_0 - x_1
        col1_b = col1_xb - t1[... , -2:-1] # 2*x_{n-1} - x_{n-2}
        col1_c = col1_xb - t1[... , -3:-2] # 2*x_{n-1} - x_{n-3}
        return [col1_a, col1_b, col1_c]
    def daub_0(self, t1) :
        ## t1: (... , z) with (z % 4) == 0
        col1_a, col1_b, _ = self.daub_cols(t1)
        s1 = tf.concat([col1_a, t1, col1_b], axis=-1) # out: (..., 2*nz)
        return s1
    def daub_1(self, t1) :
        ## t1: (... , z) with (z % 4) == 1
        col1_a, col1_b, col1_c = self.daub_cols(t1)
        s1 = tf.concat([col1_a, t1, col1_b, col1_c], axis=-1)
        return s1
    def kernel_function(self, input):
        # input: (b, x, y, c)
        # choose float precision accordingly to input
        daubechies_ker = tf.cast(self.daubechies_ker, tf.float32) if (input.dtype==tf.float32) else self.daubechies_ker
        mod_x = self.ox % 4
        mod_y = self.oy % 4
        ## pass 1: transform rows
        t1 = tf.transpose(input, perm=[0, 3, 1, 2]) # out: (b, c, ox, oy)
        if (mod_y == 0) :
            s1 = self.daub_0(t1)
        else:
            s1 = self.daub_1(t1)
        ## s1: (b, c, ox, 2*ny)
        ny_dim = s1.shape[3]
        s1 = tf.reshape(s1, [-1, ny_dim, 1])
        ## s1: (b*c*ox, 2*ny', 1)
        # build kernels and apply to rows
        k1l = tf.reshape(daubechies_ker[:,0], (4, 1, 1))
        k1h = tf.reshape(daubechies_ker[:,1], (4, 1, 1))
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        s1 = tf.concat((rl, rh), axis=-1) # out: (b*c*ox, ny, 2)
        s1 = tf.reshape(s1, [self.bs, self.cn, self.ox, self.ny, 2])
        # out: (b, c, ox, ny, 2_y)
        ## transform columns
        t2 = tf.transpose(s1, perm=[0, 1, 3, 4, 2]) # out: (b, c, ny, 2_y, ox)
        if (mod_x == 0) :
            s2 = self.daub_0(t2)
        else :
            s2 = self.daub_1(t2)
        ## s2: (b, c, ny, 2_y, 2*nx)
        nx_dim = s2.shape[4]
        s2 = tf.reshape(s2, [-1, nx_dim, 1])
        # out: (b*c*ny*2_y, 2*nx', 1)
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

class InvDaubWaveLayer2D(InvWaveLayer2D):
    """2D inverse Daubechies-N=2 trasform"""
    ########################################################################
    ## Init (with wavelet kernels)
    ########################################################################
    def __init__(self, **kwargs):
        ## Daubechies kernel
        d = math.sqrt(2) * .125 # 1/(4*sqrt(2))
        r3 = math.sqrt(3)
        h0 = d * (1+r3); h1 = d * (3+r3); h2 = d * (3-r3); h3 = d * (1-r3);
        g0 = h3; g1 = -h2; g2 = h1; g3 = -h0
        self.daubechies_ker = tf.constant([h2, h3, g2, g3, h0, h1, g0, g1], shape=(4,2), dtype=tf.float64)
        # matrix for border effect 0 (begin)
        ker_0 = tf.constant([h0, h1, h2, g0, g1, g2, 0, 0, h0, 0, 0, g0], shape=(4,3), dtype=tf.float64)
        pad_0 = tf.constant([2., -1., 1., 0., 0., 1.], shape=(3,2), dtype=tf.float64)
        self.inv_bor_0 = tf.transpose(tf.linalg.pinv(ker_0 @ pad_0), [1, 0])
        # matrix for border effect 1 (end)
        ker_1 = tf.constant([h3, 0, 0, g3, 0, 0, h1, h2, h3, g1, g2, g3], shape=(4,3), dtype=tf.float64)
        pad_1 = tf.constant([1., 0., 0., 1., -1., 2.], shape=(3,2), dtype=tf.float64)
        self.inv_bor_1 = tf.transpose(tf.linalg.pinv(ker_1 @ pad_1), [1, 0])
        ## call constructor
        super(InvWaveLayer2D, self).__init__(**kwargs)
    ########################################################################
    ## Inverse Daubechies wavelet
    ########################################################################
    def daub0_0(self, t1) :
        ## t1: (b, c, 2y, 2x) with (x % 2) == 0
        ns = t1.shape[-1] // 4 # x // 2
        d1a = t1
        d1b = t1[:,:,:,2:]
        d1b = tf.pad(d1b, [[0,0], [0,0], [0,0], [0,2]], "CONSTANT") # pad with rows (tf.stack wants same-shape tensors)
        # group by 4
        d1a = tf.reshape(d1a, [self.bs, self.cn, self.oy, ns, 4])
        d1b = tf.reshape(d1b, [self.bs, self.cn, self.oy, ns, 4])
        # out: (b, c, 2y, ns, 4)
        # stack tensors, alternating the rows
        s1 = tf.stack([d1a, d1b], axis=-2)
        s1 = tf.reshape(s1, [self.bs, self.cn, self.oy, 2*ns, 4])
        # out: (b, c, 2y, 2*ns, 4)
        s1 = s1[:,:,:,:-1,:] # remove padded rows, out: (b, c, 2y, x', 4)
        return s1
    def daub0_1(self, t1) :
        ## t1: (b, c, 2y, 2x) with (x % 2) == 1
        ns = t1.shape[-1] // 4
        d1a = t1[:,:,:,:-2]
        d1b = t1[:,:,:,2:]
        # group by 4
        d1a = tf.reshape(d1a, [self.bs, self.cn, self.oy, ns, 4])
        d1b = tf.reshape(d1b, [self.bs, self.cn, self.oy, ns, 4])
        # out: (b, c, 2y, ns, 4)
        # stack tensors, alternating the rows
        s1 = tf.stack([d1a, d1b], axis=-2)
        s1 = tf.reshape(s1, [self.bs,  self.cn, self.oy, 2*ns, 4])
        # out: (b, c, 2y, x', 4)
        return s1
    def daub1_0(self, t1) :
        ## t1: (b, c, 2x, 2y) with (y % 2) == 0
        ns = t1.shape[-1] // 4 # y // 2
        d1a = t1
        d1b = t1[:,:,:,2:]
        d1b = tf.pad(d1b, [[0,0], [0,0], [0,0], [0,2]], "CONSTANT") # pad with rows (tf.stack wants same-shape tensors)
        # group by 4
        d1a = tf.reshape(d1a, [self.bs, self.cn, self.ox, ns, 4])
        d1b = tf.reshape(d1b, [self.bs, self.cn, self.ox, ns, 4])
        # out: (b, c, 2x, ns, 4)
        # stack tensors, alternating the rows
        s1 = tf.stack([d1a, d1b], axis=-2)
        s1 = tf.reshape(s1, [self.bs, self.cn, self.ox, 2*ns, 4])
        # out: (b, c, 2x, 2*ns, 4)
        s1 = s1[:,:,:,:-1,:] # remove padded rows, out: (b, c, 2x, y', 4)
        return s1
    def daub1_1(self, t1) :
        ## t1: (b, c, 2x, 2y) with (y % 2) == 1
        ns = t1.shape[-1] // 4
        d1a = t1[:,:,:,:-2]
        d1b = t1[:,:,:,2:]
        # group by 4
        d1a = tf.reshape(d1a, [self.bs, self.cn, self.ox, ns, 4])
        d1b = tf.reshape(d1b, [self.bs, self.cn, self.ox, ns, 4])
        # out: (b, c, 2x, ns, 4)
        # stack tensors, alternating the rows
        s1 = tf.stack([d1a, d1b], axis=-2)
        s1 = tf.reshape(s1, [self.bs,  self.cn, self.ox, 2*ns, 4])
        # out: (b, c, 2x, y', 4)
        return s1
    def kernel_function(self, input):
        # input: (b, nx, ny, 4*c)
        # choose float precision accordingly to input
        daubechies_ker = tf.cast(self.daubechies_ker, tf.float32) if (input.dtype==tf.float32) else self.daubechies_ker
        inv_bor_0 = tf.cast(self.inv_bor_0, tf.float32) if (input.dtype==tf.float32) else self.inv_bor_0
        inv_bor_1 = tf.cast(self.inv_bor_1, tf.float32) if (input.dtype==tf.float32) else self.inv_bor_1
        #######################################
        ## reshape
        #######################################
        t1 = tf.reshape(input, [self.bs, self.nx, self.ny, 2, 2, self.cn])
        # out: (b, x, y, 2_y, 2_x, c)
        # apply kernel to x
        t1 = tf.transpose(t1, perm=[0, 5, 2, 3, 1, 4])
        # out: (b, c, y, 2_y, x, 2_x)
        t1 = tf.reshape(t1, [self.bs, self.cn, self.oy, self.ox])
        # out: (b, c, 2*y, 2*x)
        #######################################
        ## add borders
        #######################################
        # border 0
        b_0 = t1[:,:,:,:4]
        r2_0 = b_0 @ inv_bor_0
        # border 1
        b_1 = t1[:,:,:,-4:]
        r2_1 = b_1 @ inv_bor_1
        #######################################
        ## work on x
        #######################################
        t1 = tf.reshape(t1, [-1, self.ox, 1])
        # out: (b*c*oy, ox, 1)
        # apply kernel to x
        k1l = tf.reshape(daubechies_ker[:,0], (4, 1, 1))
        k1h = tf.reshape(daubechies_ker[:,1], (4, 1, 1))
        rl = tf.nn.conv1d(t1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(t1, k1h, stride=2, padding='VALID')
        r1 = tf.concat((rl, rh), axis=-1) # out: (b*c*oy, qx', 4)
        r1 = tf.reshape(r1, [self.bs, self.cn, self.oy, self.ox-2])
        # out: (b*c, oy, ox-2)
        #######################################
        ## merge core and borders
        #######################################
        r = tf.concat((r2_0, r1[:,:,:,1:-1], r2_1), axis=-1)
        # out: (b, c, 2*y, 2*x)
        #######################################
        ## reshape
        #######################################
        t1 = tf.transpose(r, perm=[0, 1, 3, 2])
        # out: (b, c, 2*x, 2*y)
        #######################################
        ## add borders
        #######################################
        # border 0
        b_0 = t1[:,:,:,:4]
        r2_0 = b_0 @ inv_bor_0
        # border 1
        b_1 = t1[:,:,:,-4:]
        r2_1 = b_1 @ inv_bor_1
        #######################################
        ## work on y
        #######################################
        s1 = tf.reshape(r, [self.bs, self.cn, self.oy, self.ox])
        # out: (b, c, oy, ox)
        s1 = tf.transpose(s1, perm=[0, 1, 3, 2]) # out: (b, c, ox, oy)
        s1 = tf.reshape(s1, [-1, self.oy, 1])
        # out: (b*c*ox, oy, 1)
        # apply kernel to y
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        r1 = tf.concat((rl, rh), axis=-1) # out: (b*c*ox, qy', 4)
        r1 = tf.reshape(r1, [self.bs, self.cn, self.ox, self.oy-2])
        #######################################
        ## merge core and borders
        #######################################
        r = tf.concat((r2_0, r1[:,:,:,1:-1], r2_1), axis=-1)
        # out: (b, c, 2*x, 2*y)
        r = tf.transpose(r, perm=[0, 2, 3, 1])
        # out: (b, 2*x, 2*y, c)
        return r

    
