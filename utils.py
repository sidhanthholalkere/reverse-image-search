import json
import numpy as np
import re
import string
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors

def get_caption_ids(path='data/captions_train2014.json'):
    """
    Retrieves all of the caption IDs from the COCO dataset

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    List[Int]
        The caption IDs
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    return [annotation['id'] for annotation in captions['annotations']]

def get_captions(path='data/captions_train2014.json'):
    """
    Retrieves all of the captions from the COCO dataset

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    List[String]
        The captions
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    return [annotation['caption'] for annotation in captions['annotations']]

def get_img_ids(path='data/captions_train2014.json'):
    """
    Retrieves all of the image IDs from the COCO dataset

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    List[Int]
        The image IDs
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    return list(set([annotation['image_id'] for annotation in captions['annotations']]))

def cap_id_to_im_id(path='data/captions_train2014.json'):
    """
    Returns a mapping from caption ID to image ID

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    Dict[Int -> Int]
        The key is a caption id and the value is the image id
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    return {annotation['id']: annotation['image_id'] for annotation in captions['annotations']}

def im_id_to_cap_ids(path='data/captions_train2014.json'):
    """
    Returns a mapping from an image ID to its caption IDs

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    mapping : Dict[int -> List[int]]
        The key is a caption id and the value is the image id
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    mapping = defaultdict(list)

    for annotation in captions['annotations']:
        mapping[annotation['image_id']].append(annotation['id'])

    return mapping

def im_id_to_caps(path='data/captions_train2014.json'):
    """
    Returns a mapping from an image ID to its captions

    Parameters
    ----------
    path : String
        path to the json file

    Returns
    -------
    mapping : Dict[int -> List[String]]
        The key is an image id and the value is the list of captions
    """

    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    mapping = defaultdict(list)

    for annotation in captions['annotations']:
        mapping[annotation['image_id']].append(annotation['caption'])

    return mapping

PUNC_REGEX = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return PUNC_REGEX.sub('', corpus)

def tokenize(inp):
    """
    Tokenizes a string by removing punctuation, transforming to lowercase and splitting

    Parameters
    ----------
    inp : str

    Returns
    -------
    List[str]
        the tokens
    """
    return strip_punc(inp).lower().split()

def idf(caption_path='data/captions_train2014.json'):
    """
    Returns a dictionary with key=word and value=idf

    Parameters
    ----------
    caption_path : str
        path to annotation file

    Returns
    -------
    mapping : dict[str -> float]
    """
    captions = get_captions(caption_path)
    N = len(captions)
    vocab = list(set([word for caption in captions for word in caption]))

    doc_freq = {}
    for word in vocab:
        for caption in captions:
            if word in caption:
                doc_freq[word] = doc_freq.get(word, 0) + 1

    for key in doc_freq.keys():
        doc_freq[key] = np.log10(N / doc_freq[key])

    return doc_freq

def cap_id_to_vec(caption_path='data/captions_train2014.json', w2v_path=r'data/glove.6B.50d.txt.w2v'):
    """
    Returns a mapping for a caption based on its id to its word embedding

    Parameters
    ----------
    id : int
        caption id
    caption_path : str
        path to captions
    w2v_path : str
        path to w2v file
    """
    with open(r'data/captions_train2014.json') as f:
        captions = json.load(f)

    glove = KeyedVectors.load_word2vec_format(w2v_path, binary=False) 
    freqs = idf(caption_path)

    return {annotation['id']: sum([freqs[word] * glove[word] if word in glove else 0 for word in annotation['caption']]) for annotation in captions['annotations']}
