"""A script which exposes all the datareaders."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from lucrum.datareader.binance import BinanceData

###### datareaders ########################################################
def get_data_binance(*args,**kwargs):
    """Gets historical data from binance API.
    
    Returns 
    -------
    
    """
    return BinanceData(*args, **kwargs).read()

###########################################################################
