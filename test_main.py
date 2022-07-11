"""
@Author: Conghao Wong
@Date: 2021-09-16 20:00:49
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-07 20:11:27
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import shutil

from main import main


class TestClass():
    """
    TestClass
    ---

    This class contains several test methods.
    They run the minimul training or evaluating on the key models
    to validate if codes in models or training structures run in
    the correct way.
    Note that it is not the class to validate (test) model performences.
    """

    def setup_class(self):
        if os.path.exists(p := './.test'):
            shutil.rmtree(p)

        if not os.path.exists(p := './test.log'):
            with open(p, 'w+') as f:
                f.writelines(['-----Start Test-----'])

    def teardown_class(self):
        pass

    def test_evaluate_v(self):
        self.run_with_args(['--model', 'V',
                            '--loada', './.github/workflows/test_weights/a_zara1',
                            '--loadb', 'l'])

    def run_with_args(self, args: list[str]):
        main(['null.py'] + args)


if __name__ == '__main__':
    a = TestClass()

    a.setup_class()
    a.test_evaluate_v()
    a.teardown_class()
