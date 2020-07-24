import utils
import numpy as np
from image_features import load_resnet
from cos_sim import cosine_dist

def choose_bad_img(path, good_img_caption, images):
    """

    :param path: path to resnet18 file
    :param good_img_caption: caption of the good image compared against
    :param images: shape(25, ) numpy array containing indexes of random pictures
    :return:
        descriptor vector of the image with caption closest to the good image caption
    """
    img_to_descriptor = load_resnet(path)

    img_id_to_caption = utils.im_id_to_caps()
    images = images.tolist()
    bad_img_captions = []
    for image in images:
        bad_img_captions.append(img_id_to_caption[image])

    chosen_captions = []
    for caption_cluster in bad_img_captions:
        random_indx = np.random.randint(0, len(caption_cluster))
        chosen_captions = chosen_captions.append(caption_cluster[random_indx])

    dist = np.array([cosine_dist(encoding, good_img_caption) for encoding in chosen_captions])
    img_indx = np.argmax(dist)

    return img_to_descriptor[img_indx]


def all_triplets(path):
    """
    takes in captions and creates triplets
    :params:
        path : path to resnet18 file

        model : inited model

        batch_indices : the indecies we need to return as a batch

    :return:
        List[Tuple(caption, good image, bad image)]
            List shape (300000,)
            caption : shape (50,) word embedding of an image
            good image : shape (512,) descriptor vector of image that matches caption
            bad image : shape(512,) descriptor vector of image that doesn't match caption
    
    """

    caption_id = utils.get_caption_ids()  # returns dictionary
    caption_id = caption_id[:30000]
    caption_id_to_img_id = utils.cap_id_to_im_id() #dictonary that maps caption to image ID
    img_id_to_descriptor = load_resnet(path) #dic that maps image id to descriptor
    triplets = []
    all_images = utils.get_img_ids()
    
    for indiv_caption_id in caption_id:
        caption = utils.cap_id_to_vec(indiv_caption_id)
        img_id = caption_id_to_img_id[indiv_caption_id]
        good_img = img_id_to_descriptor[img_id]
        for i in range(10):
            images = np.random.randint(0, len(all_images), size=25) #takes random array
            bad_img = choose_bad_img(path, caption, images)
            triplets.append(caption, good_img, bad_img)
    return triplets


