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
from ._haar_mm import *
from ._daubachies_mm import *

class WaveTFFactory(object) :
    """Factory for different wavelet transforms (1D/2D, haar/db2)"""
    @staticmethod
    def build(kernel_type = 'db2', dim = 2, inverse = False) :
        if (dim !=1 and dim != 2) :
            raise ValueError('Only 1- and 2-dimensional wavelet supported yet.')
        elif (kernel_type not in ['haar', 'db2']) :
            raise ValueError('Kernel type can be either "haar" or "db2".')
        # direct wavelet
        elif (inverse == False) :
            if(kernel_type == 'haar') :
                if (dim == 1) :
                    return HaarWaveLayer1D()
                else :
                    return HaarWaveLayer2D()
            elif (kernel_type == 'db2') :
                if (dim == 1) :
                    return DaubWaveLayer1D()
                else :
                    return DaubWaveLayer2D()
        # inverse wavelet
        else :
            if(kernel_type == 'haar') :
                if (dim == 1) :
                    return InvHaarWaveLayer1D()
                else :
                    return InvHaarWaveLayer2D()
            elif (kernel_type == 'db2') :
                if (dim == 1) :
                    return InvDaubWaveLayer1D()
                else :
                    return InvDaubWaveLayer2D()

