

def accuracy(Sgood, Sbad):
    """ 
    Measures the fraction of good images that are closer 
    to their embeddings than bad images

    Count up whether the similariy for the correct value  
        is the good value greater than bad, true = 1, false = 0

        mean the 1's and 0's accross the batches
        
        Parameters
        ----------
        
        Sgood - mg array, (50,)
            
        Sbad mg array, (50,)
           
        
        Returns
        -------
        Accuracy - int
            the accuracy rating of the batch
        
    """
    sum = 0
    for i in range(len(Sgood)):
        if Sgood[i] > Sbad[i]:
            sum +=1   #adds one everytime the good picture is closer to the caption than the bad caption
        #else it would add zero but I think that goes without saying
    return sum/len(Sgood) #returns the average of 0s and 1s


    