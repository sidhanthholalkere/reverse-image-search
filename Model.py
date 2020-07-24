import mygrad as mg
import mynn
import numpy as np

import utils
from cos_sim import cosine_dist
from extract_triplets import all_triplets
from accuracy import accuracy

from mygrad.nnet.losses import margin_ranking_loss

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
        
        self.dense1 = dense(dim_input, dim_output, weight_initializer=he_normal)
        
    
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
       
        return self.dense1(x)
    
       
    
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
        return self.dense1.parameters



def train(model, num_epochs, margin, path, learning_rate=0.1, batch_size=32):
    """ trains the model 
        
        Parameters
        ----------
        
        model -  Model
            an initizized Model class, with input and output dim matching the image ID(512) and the descriptor (50) 
        
        num_epochs - int
            amount of epochs
            
        margin - int
            marhine for the margine ranking loss
            
        path 
            path to the images and captions
        
        learning_rate(optional) - int
            learning rate of SDG
            
        batch_size(optional) - int
            the batch size
            

        Returns
        -------
        it trains the model by minimizing the loss function
        
        """
    optim = SGD(model.parameters, learning_rate=learning_rate)
    triplets = all_triplets(path)

    for epoch_cnt in range(num_epochs):
        images =  utils.get_img_ids(path)
        idxs = np.arange(len(images))
        np.random.shuffle(idxs)

        for batch_cnt in range(0, len(images)//batch_size):
            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            
            triplets_batch = triplets[batch_indices]

            good_pic_batch = []
            bad_pic_batch = []
            caption_batch = []

            good_pic_batch.append(i[0] for i in triplets_batch) #get the batch of pictues
            bad_pic_batch.append(i[1] for i in triplets_batch) #this one has batch_size 
            caption_batch.append(i[2] for i in triplets_batch) #batch of captions

            good_pic_pred = model(good_pic_batch)
            bad_pic_pred = model(bad_pic_batch)
            good_pic_pred = (good_pic_pred - good_pic_pred.mean()) / good_pic_pred.std()
            bad_pic_pred = (bad_pic_pred - bad_pic_pred.mean()) / bad_pic_pred.std()

            Sgood = cosine_dist(good_pic_pred, caption_batch)
            Sbad = cosine_dist(bad_pic_pred, caption_batch)
            
            loss = margin_ranking_loss(Sgood, Sbad, margin)
            acc = accuracy(Sgood, Sbad)

            loss.backward()
            optim.step()
            loss.null_gradients()

            plotter.set_train_batch({"loss" : loss.item(), "accuracy":acc}, batch_size=batch_size)

