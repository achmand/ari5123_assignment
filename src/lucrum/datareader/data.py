"""A script which exposes all the datareaders."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from lucrum.datareader.binance import BinanceData

###### datareaders ########################################################
def get_data(source, *args, **kwargs):
    sources = ["binance"]

    if source not in sources:
        error = "source=%r is not implemented" % source
        raise NotImplementedError(error)
    
    if source == "binance":
        return BinanceData(*args, **kwargs).read()

    else:
        error = "source=%r is not implemented" % source
        raise NotImplementedError(error)

def get_data_binance(*args, **kwargs):
    """Gets historical data from binance API.
    
    Returns 
    -------
    
    """
    return BinanceData(*args, **kwargs).read()

###########################################################################
