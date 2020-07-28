import pickle
from pathlib import Path
import json

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

def img_ids_to_url(ids):
    """
    Returns a list of urls based on the ids

    Parameters
    ----------
    ids : List[int]
        list of top k ids
    
    Returns
    -------
    urls : List[string]
        list of the respective url's

    """
    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)
    
    urls = {image['id']: image['coco_url'] for image in captions['images']}
    with open("idstourls", mode="wb") as opened_file:
        pickle.dump(urls, opened_file)

    return [urls[id] for id in ids]