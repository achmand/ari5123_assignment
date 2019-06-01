"""A moving average crossover strategy/algorithm."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np 
import lucrum.algo.pyta as ta 
import matplotlib.pyplot as plt
import lucrum.dataconst as dcons
from .controller import _Controller

###### moving average crossover algorithm #################################
class MaCrossoverAlgo(_Controller):   
    """An algo trading strategy which uses moving average crossovers.      
    """

    def gen_features(self, data, lead, lead_t, lag, lag_t, price=dcons.CLOSE):
        """Generates features which include the lead and lag moving averages.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices for a specific asset, expected to have OHLC.
        lead: str, 'sma', 'ema'
            Used to specify the type of moving average to be used for the lead moving average. 
            -'sma': the simple moving average
            -'ema': exponential moving average
        lead_t: integer 
            The time window used for the lead moving average. 
        lag: str, 'sma', 'ema'
            Used to specify the type of moving average to be used for the lag moving average. 
            -'sma': the simple moving average
            -'ema': exponential moving average
        price: str, optional(defaulf='close')
            The column name used to apply the lead and lag moving average on. 
            By default this will be applied to the closing price ('close') 
            but this can be applied to any column with numeric values.  
        """

        # indicators TA config 
        self.lead_col = lead + str(lead_t)
        self.lag_col = lag + str(lag_t)

        # the same type of moving average 
        if lead == lag:
            ta_config = {
                lead:[(self.lead_col, {"timeperiod":lead_t, "price":price}),
                      (self.lag_col, {"timeperiod":lag_t, "price":price})]
            }
        # different types of moving average
        else: 
            ta_config = {
                lead:[(self.lead_col, {"timeperiod":lead_t, "price":price})],
                lag:[(self.lag_col, {"timeperiod":lag_t, "price":price})]
            }
        
        # apply moving averages to dataframe
        ta.apply_ta(data, ta_config)

    def gen_positions(self, data):
        """Generates postions based on the features generated (lead/lag moving avgs).

        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices/features for a specific asset.
        """


        data["positions"] = data[self.lead_col] > data[self.lag_col]

        # map True/False to 0/1
        data["positions"] = data["positions"].map({True: 1, False: 0})

    def evaluate(self, data):
        
        # firstly calculate log returns
        data["log_returns"] = np.log(data[dcons.CLOSE]) - np.log(data[dcons.CLOSE].shift(1))
        
        # calculate profit and loss 
        data["p_l"] = data["positions"].shift(1) * data["log_returns"]

        # cumulative profit and loss 
        data["cum_p_l"] = data["p_l"].cumsum()

    def plot_perf(self, data):
        data["cum_p_l"].plot(label="Equity Curve", figsize=(15,8))
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.show()

    def stats_perf(self, data):
        cum_return = data["cum_p_l"].iloc[-1] * 100
        print("Return:{0}".format(cum_return))