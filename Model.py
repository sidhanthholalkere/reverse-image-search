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
# from load import load_file
from image_features import load_resnet

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



def train(model, num_epochs, margin, triplets, learning_rate=0.1, batch_size=32):
    """ trains the model 
        
        Parameters
        ----------
        
        model -  Model
            an initizized Model class, with input and output dim matching the image ID(512) and the descriptor (50) 
        
        num_epochs - int
            amount of epochs
            
        margin - int
            marhine for the margine ranking loss
            
        triplets 
            triplets created with the data from all_triplets(path)
        
        learning_rate(optional) - int
            learning rate of SDG
            
        batch_size(optional) - int
            the batch size
            

        Returns
        -------
        it trains the model by minimizing the loss function
        
        """
    optim = SGD(model.parameters, learning_rate=learning_rate)
    triplets = load_resnet(r"data\triplets")
    #print(triplets[0:3])
    images = utils.get_img_ids()

    for epoch_cnt in range(num_epochs):
        idxs = np.arange(len(images))
        np.random.shuffle(idxs)

        for batch_cnt in range(0, len(images)//batch_size):

            batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            triplets_batch = [triplets[index] for index in batch_indices]
            #print(triplets_batch[0])

            good_pic_batch = np.array([val[1] for val in triplets_batch])
            bad_pic_batch = np.array([val[2] for val in triplets_batch])
            caption_batch = np.array([val[0] for val in triplets_batch])

            good_pic_pred = model(good_pic_batch)
            bad_pic_pred = model(bad_pic_batch)
            good_pic_pred = good_pic_pred / mg.sqrt(mg.sum(mg.power(good_pic_pred, 2)))
            bad_pic_pred = bad_pic_pred / mg.sqrt((mg.sum(mg.power(bad_pic_pred, 2))))

            # good_pic_pred = good_pic_pred.reshape(1600, 1, 1)
            # bad_pic_pred = bad_pic_pred.reshape(1600, 1, 1)
            # caption_batch = caption_batch.reshape(1600, 1, 1)

            Sgood = (good_pic_pred * caption_batch).sum(axis=-1)
            Sbad = (bad_pic_pred * caption_batch).sum(axis=-1)
            #print(Sgood.shape, Sbad.shape)
            # Sgood = Sgood.reshape(32, 50)
            # Sbad = Sbad.reshape(32, 50)

            loss = margin_ranking_loss(Sgood, Sbad, 1, margin)
            acc = accuracy(Sgood, Sbad)
            if batch_cnt % 10 == 0:
                print(loss, acc)

            loss.backward()
            optim.step()
            loss.null_gradients()

            plotter.set_train_batch({"loss" : loss.item(), "accuracy":acc}, batch_size=batch_size)

