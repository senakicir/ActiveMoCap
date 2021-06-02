"""
Creates a wrapper around the LiftNet model.

The wrapper ensures that the model is initialized with the given weights and that it is loaded into the memory of
the GPU by running it once with an empty image.
The wrapper also receives the input data and forwards the needed inputs to the LiftNet module.
It returns the result of the prediction.
"""
import os
import numpy as np
import torch
from torch.autograd import Variable

from .liftnet import LiftNet


class LiftNetModel:
    def __init__(self, model_):
        """
        Creates a LiftNet model and loads the weights. Executes it once to make sure that it is loaded into
        memory.

        :param model_: Dictionary containing the parameters for creating the model and loading the weights.
        """
        liftnet_weight_path = os.path.join(os.path.dirname(__file__), model_['model'])
        model_dict = torch.load(liftnet_weight_path)
        model_dict = {k.replace('network.single.', ''): v for k, v in model_dict.items()}
        self.model = LiftNet(model_['nbHM'], model_['nbJoints'])
        self.model.load_state_dict(model_dict)
        self.model.cuda()
        self.model.float()
        self.model.eval()
        self.model(Variable(
                torch.from_numpy(
                        np.zeros([1, model_['nbHM'], model_['boxsize'], model_['boxsize']], dtype='float32'))).cuda())

    def forward(self, **kwargs):
        """
        Gets the input data and forwards the correct ones to the model.

        :param kwargs: The input data as a dictionary
        :return: The prediction of the model
        """
        return self.model(kwargs['heatmap'])
