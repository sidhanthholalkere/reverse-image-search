import mygrad as mg

def loss(margin, sg, sb):
    """

    :param margin: float
        margin wanted between good embeddings and bad embeddings
    :param sg:
        a good image embedding for a caption
    :param sb:
        a bad image embedding for a caption
    :return:
        the loss
    """
    return mg.maximum(0, margin - (sg - sb))
