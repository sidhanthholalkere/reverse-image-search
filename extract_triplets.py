import utils
import numpy as np
from image_features import load_resnet
from cos_sim import cosine_dist

def choose_bad_img(path, model, good_img, images):
    img_to_descriptor = load_resnet(path)
    img_descriptor = img_to_descriptor[images]
    #some way to convert descriptor into encoding
    word_encodings = model(img_descriptor)
    dist = np.array([cosine_dist(encoding, good_img) for encoding in word_encodings])
    img_indx = np.argmin(dist)
    return 








def all_triplets(path, model):
    """
    takes in captions and creates triplets
    :params:
        path : path to resnet18 file

    :return:
        triplets : List[Tuple(caption, good image, bad image)]
            List shape (300000,)
            caption : shape (50,) word embedding of an image
            good image : shape (512,) descriptor vector of image that matches caption
            bad image : shape(512,) descriptor vector of image that doesn't match caption
    """
    caption_id = utils.get_caption_ids()
    caption_id = caption_id[:30000]
    img_id = utils.get_img_ids(path)
    caption_id_to_caption = utils.cap_id_to_vec()
    caption_id_to_img_id = utils.cap_id_to_im_id()
    img_id_to_descriptor = load_resnet(path)
    for indiv_caption_id in caption_id:
        for i in range(10):
            caption = caption_id_to_caption[indiv_caption_id]
            img_id = caption_id_to_img_id[indiv_caption_id]
            good_img = img_id_to_descriptor[img_id]
            images = np.random.randint(0, len(img_id), size=25)


