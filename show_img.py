import io
from load import load_file
import requests
from PIL import Image
import image_features

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

def display_topk(ids):
    """ displays each of the top k images

    Parameters
    ----------
    ids: List[string] - shape: (k,)
        List of the top k ids
    """
    urlsdata = load_file(r"data\idstourls")
    urls = []
    for id in ids:
        urls.append(urlsdata[id])
    
    for url in urls:
        img = download_image(url)
        img.show()
