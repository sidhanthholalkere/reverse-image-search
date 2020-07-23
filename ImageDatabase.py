import pickle
from pathlib import Path

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