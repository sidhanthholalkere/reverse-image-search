import utils
import numpy as np
from image_features import load_resnet
from load import load_file
from embed_caption import se_text

def choose_bad_img(path, good_img_caption, images):
    """

    :param path: path to resnet18 file
    :param good_img_caption: caption of the good image compared against
    :param images: shape(25, ) numpy array containing indexes of random pictures
    :return:
        descriptor vector of the image with caption closest to the good image caption
    """
    img_to_descriptor = load_resnet(path)

    #img_id_to_caption = utils.im_id_to_caps()
    img_id_to_caption = load_file(r"data\imid2caps.pickle")
    bad_img_captions = []
    print("images", images)
    for image in images[0]:
        bad_img_captions.append(img_id_to_caption[image])
    #print("bad captions", bad_img_captions)
    chosen_captions = []
    for caption_cluster in bad_img_captions:
        if len(caption_cluster) > 0:
            random_indx = np.random.randint(0, len(caption_cluster))
            chosen_captions.append(caption_cluster[random_indx])
    print("chosen", chosen_captions[:10])
    bad_captions = []
    for caption in chosen_captions[:3]:
        embed = se_text(caption)
        print("embedding")
        bad_captions.append(embed)
    print(bad_captions, good_img_caption)
    dist = np.array([np.dot(encoding, good_img_caption) for encoding in bad_captions])
    img_indx = np.argmax(dist)
    print("index", img_to_descriptor[img_indx])
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
    caption_id = caption_id[:5]
    caption_id_to_img_id = utils.cap_id_to_im_id() #dictonary that maps caption to image ID
    img_id_to_descriptor = load_resnet(path) #dic that maps image id to descriptor
    triplets = []
    all_images = np.array([utils.get_img_ids()])
    #caption_id_to_caption = utils.cap_id_to_vec()
    caption_id_to_caption = load_file(r"data\capid2vec.pickle")
    for indiv_caption_id in caption_id:
        caption = caption_id_to_caption[indiv_caption_id]
        img_id = caption_id_to_img_id[indiv_caption_id]
        good_img = img_id_to_descriptor[img_id]
        print("goodimg")
        cnt = 0
        for i in range(10):
            while True:
                np.random.shuffle(all_images)
                images = all_images[cnt*25:(cnt+1)*25] #takes random array
                print(img_id, images)
                cnt += 1
                if img_id in images:
                    break
            bad_img = choose_bad_img(path, caption, images)
            triplets.append(caption, good_img, bad_img)
    return triplets


