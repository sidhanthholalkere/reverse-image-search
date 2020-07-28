import pickle
from pathlib import Path
import image_features
import Model
import numpy as np
from load import load_file
from heapq import nlargest
from collections import Counter

class ImageDatabase:
    """ A database that stores and maps the image feature
        vectors to semantic embeddings
    """

    def __init__(self):
        """Initializes a image database
        """
        # database is a dict with the image feature vectors as key
        # and semantic embeddings as values
        self.database = {}
        self.vector_id = {}
        #self.initialize_params()

    def initialize_params(self):
        """ maps the image feature vectors to 
            semantic embeddings from model
        """
        model = Model.Model(512, 50) # update this
        Model.train(model, 5, 0.1, load_file(r"data\triplets"), learning_rate=0.1, batch_size=32)
        img_vectors = image_features.load_resnet(r"data\resnet18_features.pkl")

        for key, img_vector in img_vectors.items():
            self.database[key] = model(img_vector).data

        self.vector_id = self.database

    def get_top_imgs(self, d, k=4):
        """ queries database and returns img ids
            of the top k imgs

            Parameters
            ----------
                d - word embedding shape-(50,)
                k - # of images to find
            
            Returns
            -------
                img_ids : List[]
                    list of the image ids for the top
                    k images
        """
        descriptors = np.array(list(self.database.values())).squeeze()
        cos_sim = np.dot(descriptors, d)
        part = np.argpartition(cos_sim, -k)

        ids = list(self.database.keys())
        img_ids = [ids[img] for img in part[-k:]]
        return img_ids

    def load_database(self, path):
        """
        takes in the path of the database, and returns loaded database
        ----------
            path: String
                The path of the database
        Returns
        -------
        """
        path = Path(path)
        with open(path, mode="rb") as opened_file:
            self.database = pickle.load(opened_file)

        # return 'file does not exist'

    def save_database(self, filename):
        """
        takes in the name of file you want to save to, pickles the database object and saves to that file
        ----------
            filename: String
                the name of the file
        """
        with open(filename, mode="wb") as opened_file:
            return pickle.dump(self.database, opened_file)