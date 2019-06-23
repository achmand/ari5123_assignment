"""Functions used to get financial stats and other function related with finance."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np 
from scipy.stats import skew
from scipy.stats import kurtosis

###### financial functions ################################################
def dist_moments(x):
    """Calculate the four distribution moments. 
        
    Parameters 
    ----------
    x (numpy array): 
        The array used to calculate the distribution moments for. 
        
    Returns
    -------
    numpy array: 
        1st distribution moment which is the mean of the array.
    numpy array: 
        2nd distribution moment which is the standard deviation of the array.
    numpy array: 
        3rd distribution moment which is the skew of the array.
    numpy array: 
        4th distribution moment which is the kurtosis of the array. 
    """

    return np.mean(x), np.std(x), skew(x), kurtosis(x)

###########################################################################
