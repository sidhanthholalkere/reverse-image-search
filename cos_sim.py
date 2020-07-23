import numpy as np

def cosine_dist(d1, d2):
    """
    takes 2 vectors shape (50,) and returns cosine distance
    Parameters
    ----------
    d1 : numpy array
        first d vector
    d2 : numpy array    
        2nd d vector
    Returns
    -------
    float
        cosine distance of d1 and d2
    
    """
    return 1 - (np.dot(d1, d2)) / (np.linalg.norm(d1)* np.linalg.norm(d2))


import io
import requests
from PIL import Image

__all__ = ["download_image"]


def download_image(img_url: str) -> Image:
    """ Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))