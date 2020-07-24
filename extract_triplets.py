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


def all_triplets(path, model, batch_indices):
    """
    takes in captions and creates triplets
    :params:
        path : path to resnet18 file

        model : inited model

        batch_indices : the indecies we need to return as a batch

    :return:
        triplets : Tuple(good image batch, bad image batch, caption batch)
            arrays of size batch_size, 512 for the images and batch_size, 50 for captions
    
    """
    caption_id = utils.get_caption_ids(path) #returns dictionary
    caption_id_batch = caption_id[batch_indices] #take batch idecies for captions

    img_id = utils.get_img_ids(path) #list of image ids
    img_id_batch = img_id[batch_indices] #take a batch of image ids

    bad_image_batch = choose_bad_img(path, model, img_id_batch, img_id)

    return img_id_batch, bad_image_batch, caption_id_batch


    """
    not sure what's happening here and I'm a little afraid to touch it. 
    We don't need these dictionaries, and captions for this function, right?

    caption_id_to_caption = utils.cap_id_to_vec() #caption IDs to captions
    caption_id_to_img_id = utils.cap_id_to_im_id() #dictonary that maps caption to image ID
    img_id_to_descriptor = load_resnet(path) #dic that maps image id to descriptor 
    
    for indiv_caption_id in caption_id:
        for i in range(10):
            caption = caption_id_to_caption[indiv_caption_id]
            img_id = caption_id_to_img_id[indiv_caption_id]
            good_img = img_id_to_descriptor[img_id]
            images = np.random.randint(0, len(img_id), size=25) #takes random array


    old return statment in case I made a mistake when I restucted the tuple
    List[Tuple(caption, good image, bad image)]
            List shape (300000,)
            caption : shape (50,) word embedding of an image
            good image : shape (512,) descriptor vector of image that matches caption
            bad image : shape(512,) descriptor vector of image that doesn't match caption


    """
