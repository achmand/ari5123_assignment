"""Class which interacts with Binance API (data retrieval)."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np 
import pandas as pd
import lucrum.dataconst as dcons
from binance.client import Client 
from lucrum.datareader.base import BaseData

###### binance data api class #############################################
class BinanceData(BaseData):
    """Gets historical cryptocurrency data from Binance API.      
    """

    # columns used when getting historical data 
    _hist_cols = [dcons.OPEN_TIME, 
                  dcons.OPEN,
                  dcons.HIGH,
                  dcons.LOW,
                  dcons.CLOSE,
                  dcons.VOLUME,
                  dcons.CLOSE_TIME,
                  "quote_asset_volume",
                  dcons.TRADES,
                  "buy_base_volume",
                  "buy_quote_volume",
                  "ignore"]

    def __init__(self, symbols, start, end, interval, limit=500, timezone=None, 
                api_key="api-key", api_secret="api-secret", requests_params=None):
   
        """Binance data reader constructor.
        
        Parameters 
        ----------
        symbols : str, List[str]
            List or single symbol used when retrieving data from some datareader. 
        start: str 
            The start date to fetch data from. 
        end: str 
            The end date to fetch data from. 
        interval : str  
            The interval used when fetching the data. 
        limit : integer, optional 
            By default set to 500, max is 1000. 
        timezone : str, optional, 
            Dates returned by datasource are in UTC, if specified
            dates are converted to the timezone specified. 
        api_key : str, optional
            The API key used to interact with the binance API. 
            Certain API calls require an API key.
        api_secret : str, optional 
            The API secrect to interact with the binance API. 
            Certain API calls require an API secret.
        requests_params : dictionary, optional 
            Dictionary of request params to use for all the API calls. 
        
        Attributes
        ----------
        symbols : str, List[str]
            List or single symbol used when retrieving data from some datareader. 
        start: str 
            The start date to fetch data from. 
        end: str 
            The end date to fetch data from. 
        interval : str  
            The interval used when fetching the data. 
        limit : integer 
            By default set to 500, max is 1000. 
        _client : instance Client
            An instance of Client from binance.client found in the 
            python-binance library which is used as an interface to 
            interact with the binance API. 

        Notes
        -----
        This class utilizes the python-binance library for more information
        about the parameters used to initialize the client (e.g: requests_params) 
        visit the following link: https://python-binance.readthedocs.io/en/latest/overview.html#initialise-the-client.
        """    

        # call base constructor 
        super(BinanceData, self).__init__(symbols, start, end)

        # set attributes from parameters
        self.interval = interval
        self.limit = limit
        self.timezone = timezone

        # create a new instance of binance.client "Client" and set to attribute    
        self._client = Client(api_key, api_secret, requests_params)

    def close(self):
        """In this datareader there are no streams/sessions."""
        pass

    def read_datareader(self):
        """Gets historical data from binance API.

        Returns
        -------
        """

        # gets the historical prices from the _client 
        hist_prices = self._client.get_historical_klines(symbol=self.symbols,
                                                        interval=self.interval, 
                                                        start_str=self.start,
                                                        end_str=self.end, 
                                                        limit=self.limit)

        # convert historical prices to pandas dataframe 
        hist_prices = pd.DataFrame(data = np.array(hist_prices),
                                  columns=self._hist_cols)

        # since the datetime is in unix format, 
        # it needs to be converted to datetime 
        hist_prices[dcons.OPEN_TIME] = pd.to_datetime(hist_prices[dcons.OPEN_TIME], unit="ms")
        hist_prices[dcons.CLOSE_TIME] = pd.to_datetime(hist_prices[dcons.CLOSE_TIME], unit="ms")

        # timezone returned from binance is in UTC, 
        # if specified convert to timezone 
        if self.timezone is not None:
            hist_prices[dcons.OPEN_TIME] = hist_prices[dcons.OPEN_TIME].dt.tz_localize('UTC').dt.tz_convert(self.timezone)
            hist_prices[dcons.CLOSE_TIME] = hist_prices[dcons.CLOSE_TIME].dt.tz_localize('UTC').dt.tz_convert(self.timezone)

        # convert any columns which are floats
        hist_prices[[dcons.OPEN, dcons.HIGH, dcons.LOW, dcons.CLOSE, dcons.VOLUME, 
                    "quote_asset_volume", "buy_base_volume", "buy_quote_volume"]] = hist_prices[[dcons.OPEN, dcons.HIGH, 
                                                                                                dcons.LOW, dcons.CLOSE, 
                                                                                                dcons.VOLUME, "quote_asset_volume", 
                                                                                                "buy_base_volume", "buy_quote_volume"]].astype(float)
        # convert any columns which are integers 
        hist_prices[dcons.TRADES] = hist_prices[dcons.TRADES].astype(int)

        # return historical prices as a pandas dataframe
        return hist_prices[dcons.OHLC_COLS]

###########################################################################
