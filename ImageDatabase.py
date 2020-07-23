import pickle
from pathlib import Path
import image_features
import Model

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

    def initialize_params(self):
        """ maps the image feature vectors to 
            semantic embeddings from model
        """
        model = Model.Model(512, 50) # update this
        img_vectors = image_features.load_resnet("data/resnet18_features.pkl")

        for img_vector in img_vectors.values():
            self.database[img_vector] = model(img_vector)

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