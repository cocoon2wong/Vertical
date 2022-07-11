"""
@Author: Conghao Wong
@Date: 2021-07-19 11:11:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-22 19:59:57
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

BASE_DIR = './logs'

for d in os.listdir(BASE_DIR):
    if d.startswith('.') or not os.path.isdir(d):
        continue
    
    files = os.listdir(cd := os.path.join(BASE_DIR, d))
    
    if (fn := 'best_ade_epoch.txt') in files:
        best_epoch = np.loadtxt(os.path.join(cd, fn))[1].astype(int)
        pattern = '_epoch{}.tf'.format(best_epoch)
    
    else:
        continue

    for f in files:
        path = os.path.join(cd, f)
        if pattern in f:
            print('Find {}.'.format(path))

        else:
            if f.endswith('.tf.index') or '.tf.data' in f:
                print('Remove {}.'.format(path))
                os.remove(path)
            
