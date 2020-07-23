import utils

def all_triplets():
    """
    takes in captions and creates triplets
    :return:
        triplets : Tuple(caption, good image, bad image)
            caption : shape (50,) word embedding of an image
            good image : shape (512,) descriptor vector of image that matches caption
            bad image : shape(512,) descriptor vector of image that doesn't match caption
    """
    caption_id = utils.get_caption_ids()
    caption_id = caption_id[:30000]
    caption_id_to_caption = utils.cap_id_to_vec()
    caption_id_to_img_id = utils.cap_id_to_im_id()
    for indiv_caption_id in caption_id:
        for i in range(10):
            caption = caption_id_to_caption[indiv_caption_id]
            img_id =  caption_id_to_img_id[indiv_caption_id]
