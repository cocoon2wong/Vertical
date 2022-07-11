"""
@Author: Conghao Wong
@Date: 2022-06-20 20:10:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-20 20:10:58
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os


def dir_check(target_dir: str) -> str:
    """
    Used for check if the `target_dir` exists.
    It not exist, it will make it.
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    return target_dir