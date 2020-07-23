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

def display_top4(urls):
    """ displays each of the top 4 images

    Parameters
    ----------
    urls: List[string] - shape: (4,)
        List of the top 4 urls
    """
    for url in urls:
        img = download_image(url)
        img.show()