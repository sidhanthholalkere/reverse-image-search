import pickle
from pathlib import Path
import image_features
import Model
import cos_sim
import numpy as np
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
        self.initialize_params()

    def initialize_params(self):
        """ maps the image feature vectors to 
            semantic embeddings from model
        """
        model = Model.Model(512, 50) # update this
        img_vectors = image_features.load_resnet("data/resnet18_features.pkl")

        for img_vector in img_vectors:
            self.database[img_vector] = model(img_vector)
        
        self.vector_id = {img_embed: img_id for img_id, img_embed in img_vectors.items()}

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
        cos_dis = {}
        for img_d in self.database.values():
            cos_dis[img_d] = cos_sim.cosine_dist(d, img_d)
        
        count_dis = Counter(cos_dis)
        k_img_embs = [img_emb for img_emb, cosd in count_dis.most_common(k)]

        img_ids = [self.vector_id[img] for img in k_img_embs]

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