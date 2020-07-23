import json
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

    return {annotation['id']: glove[annotation['caption']] for annotation in captions['annotations']}
