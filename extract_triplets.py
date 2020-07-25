import utils
import numpy as np
from image_features import load_resnet
import pickle
from load import load_file

def choose_bad_img(path, good_img_caption, images, cap2vec, imgid2capid):
    """

    :param path: path to resnet18 file
    :param good_img_caption: caption of the good image compared against
    :param images: shape(25, ) numpy array containing indexes of random pictures
    :return:
        descriptor vector of the image with caption closest to the good image caption
    """

    img_id_to_caption = imgid2capid
    #img_id_to_caption = load_file(r"data\imid2caps.pickle")
    #print("images", images)
    bad_img_captions = [img_id_to_caption[image] for image in images[0]]
    #print("bad captions", bad_img_captions)
    chosen_captions = []
    for caption_cluster in bad_img_captions:
        random_indx = np.random.randint(0, len(caption_cluster))
        chosen_captions.append(caption_cluster[random_indx])
    #print("chosen", chosen_captions)
    caption_to_vector = cap2vec
    #print(sum(isinstance(caption_to_vector[caption], int) for caption in chosen_captions))
    bad_captions = [caption_to_vector[caption] for caption in chosen_captions]
    #print("bad", bad_captions)
    #print(bad_captions, good_img_caption)
    #print(caption_to_vector[chosen_captions[1]])
    dist = np.array([np.dot(encoding, good_img_caption) for encoding in bad_captions])
    #print(dist, dist.shape)
    img_indx = np.argmax(dist)
    #print("index", images[0][img_indx])
    return images[0][img_indx]


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
    img_id_to_caption_id = utils.im_id_to_cap_ids()
    triplets = []
    all_images = np.array([utils.get_img_ids()])
    #caption_id_to_caption = utils.cap_id_to_vec()
    #with open("capid2cap", mode="wb") as opened_file:
    #    pickle.dump(caption_id_to_caption, opened_file)

    caption_id_to_caption = load_file(r"data\capid2cap")
    for i, indiv_caption_id in enumerate(caption_id[2128:]):
        print(i)
        caption = caption_id_to_caption[indiv_caption_id]
        img_id = caption_id_to_img_id[indiv_caption_id]
        if img_id in img_id_to_descriptor.keys():
            good_img = img_id_to_descriptor[img_id]
        else:
            continue
        #print("goodimg")
        for i in range(10):
            while True:
                rng = np.random.default_rng()
                images = rng.choice(all_images, size=25, axis=1)
                #print(images)
                #print(img_id, images.shape, all_images.shape)
                if img_id not in images:
                    break
            img_key = choose_bad_img(path, caption, images, caption_id_to_caption, img_id_to_caption_id)
            if img_key in img_id_to_descriptor.keys():
                bad_img = img_id_to_descriptor[img_key]
            #print(bad_img)
            triplets.append((caption, good_img, bad_img))
    with open("triplets", mode="wb") as opened_file:
        pickle.dump(triplets, opened_file)
    return triplets


