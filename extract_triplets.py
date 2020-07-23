import utils
import numpy as np
from image_features import load_resnet
from cos_sim import cosine_dist

def choose_bad_img(path, model, good_img):
    img_to_descriptor = load_resnet(path)
    indx = np.arange(300)
    np.random.shuffle(indx)
    for batch_cnt in range(len(indx) // 30):
        batch_indx = indx[batch_cnt*30:(batch_cnt+1)*30]
        batch = img_to_descriptor[batch_indx]
        word_embeddings = model(batch)
        dist = cosine_dist()






def all_triplets(path):
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
    caption_id_to_caption = utils.cap_id_to_vec()
    caption_id_to_img_id = utils.cap_id_to_im_id()
    img_id_to_descriptor = load_resnet(path)
    for indiv_caption_id in caption_id:
        for i in range(10):
            caption = caption_id_to_caption[indiv_caption_id]
            img_id = caption_id_to_img_id[indiv_caption_id]
            good_img = img_id_to_descriptor[img_id]


