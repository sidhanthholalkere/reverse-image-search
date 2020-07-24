import numpy as np
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors
import re, string
import utils


def se_text(query):
    """
    loads Glove50 embeddings, weights them according to IDF
    Parameters
    ----------
    query : str
        caption we want to match image to

    Returns
    -------
    weighted embeddings

    """
    path = r"./glove.6B.50d.txt.w2v"
    glove = KeyedVectors.load_word2vec_format(path, binary=False)

    tokens = strip_punc(query).lower().split()
    weighted_embeddings = np.array([glove[token] for token in tokens])
    idf = get_idf(query)
    return weighted_embeddings * idf


def strip_punc(s):
    """
    removes punctuation from String s
    """
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub('', s)


def get_idf(query):
    """
    gets the idf values for all words in String query
    """
    all_captions = utils.get_captions("data/captions_train2014.json")
    all_word_cnt = [to_counter(cap) for cap in all_captions]    # list of counter of words in each caption
    vocab = sorted(strip_punc(query).lower().split())
    N = len(all_word_cnt)
    nt = [sum(1 if t in counter else 0 for counter in all_word_cnt) for t in vocab]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)


def to_counter(caption):
    return Counter(strip_punc(caption).lower().split())


def to_vocab(counters):
    vocab = set()
    for counter in counters:
        vocab.update(counter)
    return sorted(vocab)
