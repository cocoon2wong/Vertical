"""
@Author: Conghao Wong
@Date: 2022-06-23 10:45:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:22:06
@Description: Structure for the V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from silverballers import BaseSilverballers

from .__Valpha import VAModel
from .__Vbeta import VBModel


class V(BaseSilverballers):
    """
    V^2-Net
    ---

    Training structure for the V^2-Net model.
    It has keypoints-interpolation two sub-networks.
    Both these sub-networks implement on agents' trajectory spectrums.
    """

    def __init__(self, terminal_args: list[str]):

        self.set_models(agentModel=VAModel,
                        handlerModel=VBModel)
                        
        super().__init__(terminal_args)
