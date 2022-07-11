"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-07 20:09:22
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import codes as C
import vertical


def main(args: list[str]):
    
    _args = C.args.BaseArgTable(terminal_args=args)
    model = _args.model

    # ---------------
    # Vertical models
    # ---------------
    if model in ['va', 'agent']:
        s = vertical.VA

    elif model == 'vb':
        s = vertical.VB

    elif model == 'V':
        s = vertical.V

    elif model == 'mv':
        s = vertical.MinimalV
    
    else:
        raise NotImplementedError(
            'model type `{}` is not supported.'.format(model))

    s(terminal_args=args).train_or_test()


if __name__ == '__main__':
    main(sys.argv)
    
    