import json
import numpy as np
import re
import string
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import pickle
from pathlib import Path


class Utils_Class:
    def __init__(self):
        self.imid2caps = {}
        self.idfs = {}
        self.capid2vecs = {}
        self.capid2imid = {}
        self.PUNC_REGEX = re.compile('[{}]'.format(re.escape(string.punctuation)))

    def get_caption_ids(self, path='data/captions_train2014.json'):
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

    def get_captions(self, path='data/captions_train2014.json'):
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

    def get_img_ids(self, path='data/captions_train2014.json'):
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

    def cap_id_to_im_id(self, path='data/captions_train2014.json'):
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

        self.capid2imid = {annotation['id']: annotation['image_id'] for annotation in captions['annotations']}
        return self.capid2imid

    def im_id_to_cap_ids(self, path='data/captions_train2014.json'):
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

    def im_id_to_caps(self, path='data/captions_train2014.json'):
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

        self.imid2caps = mapping
        return mapping

    def strip_punc(self, corpus):
        """ Removes all punctuation from a string.

            Parameters
            ----------
            corpus : str

            Returns
            -------
            str
                the corpus with all punctuation removed"""
        # substitute all punctuation marks with ""
        return self.PUNC_REGEX.sub('', corpus)

    def tokenize(self, inp):
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
        return self.strip_punc(inp).lower().split()

    def idf(self, caption_path='data/captions_train2014.json'):
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
        captions = self.get_captions(caption_path)
        N = len(captions)
        vocab = list(set([word for caption in captions for word in caption]))

        doc_freq = {}
        for word in vocab:
            for caption in captions:
                if word in caption:
                    doc_freq[word] = doc_freq.get(word, 0) + 1

        for key in doc_freq.keys():
            doc_freq[key] = np.log10(N / doc_freq[key])

        self.idfs = doc_freq
        return doc_freq

    def cap_id_to_vec(self, caption_path='data/captions_train2014.json', w2v_path=r'data/glove.6B.50d.txt.w2v'):
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
        freqs = self.idf(caption_path)

        self.capid2vecs = {annotation['id']: sum([freqs[word] * glove[word] if word in glove else 0 for word in annotation['caption']]) for annotation in captions['annotations']}
        return self.capid2vecs

    def save_mappings(self, imid2caps_file, idfs_file, capid2vecs_file, capid2imid_file):
        with open(imid2caps_file, mode="wb") as opened_file:
            pickle.dump(self.imid2caps, opened_file)
        with open(idfs_file, mode="wb") as opened_file:
            pickle.dump(self.idfs, opened_file)
        with open(capid2vecs_file, mode="wb") as opened_file:
            pickle.dump(self.capid2vecs, opened_file)
        with open(capid2imid_file, mode="wb") as opened_file:
            pickle.dump(self.capid2imid, opened_file)

    def load_mappings(self, imid2caps_path, idfs_path, capid2vecs_path, capid2imid_path):
        imid2caps_path = Path(imid2caps_path)
        with open(imid2caps_path, mode="rb") as opened_file:
            self.imid2caps = pickle.load(opened_file)

        idfs_path = Path(idfs_path)
        with open(idfs_path, mode="rb") as opened_file:
            self.idfs = pickle.load(opened_file)

        capid2vecs_path = Path(capid2vecs_path)
        with open(capid2vecs_path, mode="rb") as opened_file:
            self.capid2vecs = pickle.load(opened_file)

        capid2imid_path = Path(capid2imid_path)
        with open(capid2imid_path, mode="rb") as opened_file:
            self.capid2imid = pickle.load(opened_file)
