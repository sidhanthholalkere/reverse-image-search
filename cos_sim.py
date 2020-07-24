import mygrad as mg

def cosine_dist(d1, d2):
    """
    takes 2 vectors shape (50,) and returns cosine distance
    Parameters
    ----------
    d1 : numpy array
        first d vector
    d2 : numpy array    
        2nd d vector
    Returns
    -------
    float
        cosine distance of d1 and d2
    
    """
    return mg.matmul(d1, d2)