"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 14:33:11
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import codes as C
import tensorflow as tf

from .__args import SilverballersArgs
from .agents.__baseAgent import BaseAgentStructure
from .handlers.__baseHandler import BaseHandlerModel, BaseHandlerStructure


class BaseSilverballersModel(C.basemodels.Model):

    def __init__(self, Args: SilverballersArgs,
                 agentModel: C.basemodels.Model,
                 handlerModel: BaseHandlerModel = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.set_preprocess()

        self.agent = agentModel
        self.handler = handlerModel
        self.linear = not self.handler

        if self.linear:
            self.linear_layer = C.basemodels.layers.LinearInterpolation()

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None,
             *args, **kwargs):

        outputs = self.agent.forward(inputs)

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        pos = self.agent.structure.p_index

        # shape = (batch, Kc, n, 2)
        proposals = outputs[0]
        current_inputs = inputs

        if self.linear:
            # Piecewise linear interpolation
            pos = tf.cast(pos, tf.float32)
            pos = tf.concat([[-1], pos], axis=0)
            obs = current_inputs[0][:, tf.newaxis, -1:, :]
            proposals = tf.concat([tf.repeat(obs, Kc, 1), proposals], axis=-2)

            final_results = self.linear_layer.call(index=pos, value=proposals)

        else:
            # call the second stage model
            handler_inputs = [inp for inp in current_inputs]
            handler_inputs.append(proposals)
            final_results = self.handler.forward(handler_inputs)[0]

        return (final_results,)


class BaseSilverballers(C.training.Structure):

    """
    Basic structure to run the `agent-handler` based silverballers model.
    Please set agent model and handler model used in this silverballers by
    subclassing this class, and call the `set_models` method *before*
    the `super().__init__()` method.
    """

    # Structures
    agent_structure = BaseAgentStructure
    handler_structure = BaseHandlerStructure

    # Models
    agent_model = None
    handler_model = None
    silverballer_model = BaseSilverballersModel

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        # set args
        self.args = SilverballersArgs(terminal_args)
        self.important_args += ['K']

        # set inputs and outputs
        self.set_inputs('trajs', 'maps', 'paras')
        self.set_labels('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # check weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`Agent` or `Handler` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        # assign models
        self.agent = self.agent_structure(
            terminal_args + ['--load', self.args.loada])
        self.agent.set_model_type(self.agent_model)
        self.agent.load_best_model(self.args.loada)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True

        else:
            self.linear_predict = False
            self.handler = self.handler_structure(
                terminal_args + ['--load', self.args.loadb])
            self.handler.set_model_type(self.handler_model)
            self.handler.args._set('key_points', self.agent.args.key_points)
            self.handler.load_best_model(self.args.loadb, asHandler=True)

        if self.args.batch_size > self.agent.args.batch_size:
            self.args._set('batch_size', self.agent.args.batch_size)
        self.args._set('test_set', self.agent.args.test_set)

    def set_models(self, agentModel: type[C.basemodels.Model],
                   handlerModel: type[BaseHandlerModel],
                   agentStructure: type[BaseAgentStructure] = None,
                   handlerStructure: type[BaseHandlerStructure] = None):
        """
        Set models and structures used in this silverballers instance.
        Please call this method before the `__init__` method when subclassing.
        You should better set `agentModel` and `handlerModel` rather than
        their training structures if you do not subclass these structures.
        """
        if agentModel:
            self.agent_model = agentModel

        if agentStructure:
            self.agent_structure = agentStructure

        if handlerModel:
            self.handler_model = handlerModel

        if handlerStructure:
            self.handler_structure = handlerStructure

    def train_or_test(self):
        self.model = self.create_model()
        self.run_test()

    def create_model(self, *args, **kwargs):
        return self.silverballer_model(
            self.args,
            agentModel=self.agent.model,
            handlerModel=None if self.linear_predict else self.handler.model,
            structure=self,
            *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], dataset: str):
        """
        Information to show (or to log into files) after testing
        """
        self.print_parameters(title='test results',
                              dataset=dataset,
                              **loss_dict)
        self.log('Results from {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict))
