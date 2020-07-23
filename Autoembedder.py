import numpy as np
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal

class Autoembedder:
    def __init__(self, d_in, d_out):
        """

        :param d_in: int dimensions in
        :param d_out: int dimensions out
        """
        self.dense1 = dense(d_in, d_out, weight_initializer=glorot_normal)

    def __call__(self, x):
        """

        :param x: shape (1, 512) image descriptor
        :return: shape (1,50) embedding for the image
        """
        return self.dense1(x)

    @property
    def parameters(self):
        return self.dense1.parameters