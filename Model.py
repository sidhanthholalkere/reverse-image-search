import mygrad as mg
import mynn
import numpy as np

from mygrad.nnet.initializers import he_normal
from mynn.layers.dense import dense

from mynn.optimizers.sgd import SGD

import matplotlib.pyplot as plt


class Model():
    def __init__(self, dim_input, dim_output): #default dim_input should be 512, output should be 50
        """ Initializes all layers needed for RNN
        
        Parameters
        ----------
        dim_input: int
            Dimensionality of data passed to RNN
        
        
        dim_output: int
            Dimensionality of output of RNN
        """
        
        dense1 = dense(dim_input, dim_output, weight_initializer=he_normal)
        
    
    def __call__(self, x):
        """ Performs the full forward pass for the RNN.
                
        Parameters
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(dim_input,)
            og image embeding
            
        Returns
        -------
        mygrad.Tensor, shape=(dim_output,)
            compressed image embeding
        """
       
        return dense1(x)
    
       
    
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        """
        # STUDENT CODE HERE
        return dense1.params

def loss(Sgood, Sbad, margin):
        """ computes the Margin Ranking Loss 
        
        Parameters
        ----------
        
        Sgood - mg array, (50,)
            the cos similarity between image and good caption
            
        Sbad mg array, (50,)
            the cos similarity between the image and the bad caption

        margin - int
            the amount Sgood should be better than Sbad
        
        Returns
        -------
        Loss - mg array, (50,)
            the margin ranking loss (0 if Sgood is bigger than Sbad + magrin, a linear loss otherwise)
        
        """
def accuracy():
    """ Count up whether the similariy for the correct value  
        is the good value greater than bad, true = 1, false = 0

        mean the 1's and 0's accross the batches
        
        Parameters
        ----------
        
        Sgood - mg array, (50,)
            the cos similarity between image and good caption
            
        Sbad mg array, (50,)
            the cos similarity between the image and the bad caption
        
        Returns
        -------
        Accuracy - int
            the accuracy rating of the batch
        
        """



se_image = Model(512, 50)

optim = SGD(se_image.parameters, learning_rate=10e-3)