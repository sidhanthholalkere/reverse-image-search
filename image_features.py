import pickle
from pathlib import Path

def load_resnet(path):
        """
        takes in the path of the resnet18 features file and returns the image 
        vectors
        ----------
            path: String
                The path of the resnet18 features file
        Returns
        -------
            features: dict
                maps image id to image feature vector
        """
        path = Path(path)
        with open(path, mode="rb") as opened_file:
            return pickle.load(opened_file)