"""A combination of all indicators strategy/algorithm."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import lucrum.algo.pyta as ta 
import matplotlib.pyplot as plt
import lucrum.dataconst as dcons
import lucrum.algo.finstats as fs
from .controller import _Controller

###### combined TA strategy class #########################################
class CombinedTaAlgo(_Controller):
    """A combination of all indicators trading strategy, includes EMA,SMA,RSI,William%R and NATR.      
    """

    def gen_features(self, 
                     data, 
                     ema_lead, 
                     ema_lag, 
                     sma_lead, 
                     sma_lag, 
                     rsi_window, 
                     willr_window,
                     natr_window, 
                     price=dcons.CLOSE):
        """Generates features for all the combined technical indicators.
           Which include EMA, SMA, RSI and William%R. It also included NATR. 
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices for a specific asset, expected to have OHLC.
        ema_lead: integer
            The time window used for the EMA lead.
        ema_lag: integer
            The time window used for the EMA lag.
        sma_lead: integer
            The time window used for the SMA lead.
        sma_lag: integer
            The time window used for the SMA lag.  
        rsi_window: integer 
            The time window used for the RSI. 
        willr_window:
            The time window used for William%R.
        natr_window:
            The time window used for NATR. 
        price: str, optional(defaulf='close')
            The column name used to apply the lead and lag moving average on. 
            By default this will be applied to the closing price ('close') 
            but this can be applied to any column with numeric values.  
        """

        # set config for ta-lib
        ta_config = {
            "ema":[("lead_ema", {"timeperiod":ema_lead, "price":price}),
                  ("lag_ema", {"timeperiod":ema_lag, "price":price})],
            "sma":[("lead_sma", {"timeperiod":sma_lead, "price":price}),
                  ("lag_sma", {"timeperiod":sma_lag, "price":price})],
            "rsi":[("rsi", {"timeperiod": rsi_window, "price": price})],
            "willr":[("willr", {"timeperiod":willr_window})],
            "natr":[("natr", {"timeperiod":natr_window})]
        }

        # apply RSI to dataframe
        ta.apply_ta(data, ta_config)

        
        
    def gen_positions(self, 
                      data, 
                      rsi_overbought, 
                      rsi_oversold, 
                      willr_overbought, 
                      willr_oversold,
                      volatility):
        """Generates postions based on all the TA indicatiors.
           This generates both short and long positions. 

        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices/features for a specific asset.
        rsi_oversold: int
            The upper bound which indicates an overbought (> 1 and < 100).
        rsi_oversold: int
            The lower bound which indicates an oversold (> 1 and < 100) must be smaller than upper.
        willr_overbought: int
            The value which indicates an overbought. If the reading is above this value. 
        willr_oversold: int
            The value which indicates an oversold. If the reading is below this value. 
        volatility: float
            The threshold used by NATR to give a signal.
        """
        
        # generates signals 
        data["ema_long_signal"] = data["lead_ema"] > data["lag_ema"]   # EMA long signal 
        data["ema_short_signal"] = data["lead_ema"] <= data["lag_ema"] # EMA short signal
        data["sma_long_signal"] = data["lead_sma"] > data["lag_sma"]   # SMA long signal 
        data["sma_short_signal"] = data["lead_sma"] <= data["lag_sma"] # SMA short signal
        data["rsi_long_signal"] = data["rsi"] < rsi_oversold           # RSI long signal 
        data["rsi_short_signal"] = data["rsi"] > rsi_overbought        # RSI short signal
        data["will_r_long_signal"] = data["willr"] < willr_oversold    # WillR long signal
        data["will_r_short_signal"] = data["willr"] > willr_overbought # WillR short signal
        data["natr_signal"] = data["natr"] > volatility # above a volatility threshold
        
        # long signal 
        data["long_signal"] = ((((data["ema_short_signal"] == 1) & (data["sma_long_signal"] == 1)) 
                               | ((data["rsi_long_signal"] == 1) & (data["will_r_long_signal"] == 1)))
                               & (data["natr_signal"] == 1))
        
        # short signal 
        data["short_signal"] = ((((data["ema_long_signal"] == 1) & (data["sma_short_signal"] == 1)) 
                               | ((data["rsi_short_signal"] == 1) & (data["will_r_short_signal"] == 1)))
                               & (data["natr_signal"] == 1))
        
        # generate positions 
        data["long_position"] = 0  # initially set to 0 
        data["short_position"] = 0 # initially set to 0
        data.loc[data.long_signal, "long_position"] = 1    # set position to go long
        data.loc[data.short_signal, "short_position"] = -1 # set position to short 

        # set position at time t 
        data["position"] = data.long_position + data.short_position

        # basically we check for any changes between positions 
        # except when we are not in a trade at all (position is equal to 0)
        data["apply_fee"] = 0
        data["apply_fee"] = ((data.position != data.position.shift(-1)) | (data.position != data.position.shift(1))).astype(int)
        data.loc[data.position == 0, "apply_fee"] = 0  
        
        # now since we can close a position and go on sit/hold 
        # we need to apply a fee when we exited position and the next position is 0 
        # this happens when you get in a position and exit after the next interval 
        data.loc[(data.position == 0) & (data.apply_fee.shift(1) == 1) & (data.position.shift(1) != data.position.shift(2)) , "apply_fee"] = 1  

    def evaluate(self, data, trading_fee):
        """Evaluates/calculates performance from the positions generated.  

        Parameters 
        ----------
        data: pandas dataframe 
            Holds positions generated when apply the gen_positions function.
        trading_fee: float
            The trading fee applied which each trade. 
        """

        # firstly calculate log returns
        data["logprices"] = np.log(data[dcons.CLOSE])
        data["log_returns"] = data.logprices - data.logprices.shift(1)
        
        # calculate profit and loss 
        data["pl"] = data["position"].shift(1) * data["log_returns"]

        # new we need to apply fee in profit and loss when we entered or exited a trade
        # first we get index where a fee is suppose to be applied 
        fee_indices = data.loc[data.apply_fee == 1].index.values + 1

        # must check index does not go beyond shape 
        # special case when last row is a trade 
        fee_indices = fee_indices[fee_indices < data.shape[0]]

        # now we apply fee to the p/l at location where the trade was made + 1
        data.loc[fee_indices , "pl"] =  (1 - trading_fee) * data["pl"]

        # cumulative profit and loss 
        data["cum_pl"] = data["pl"].cumsum()

    def plot_pos(self, data):
        
        # figure size
        fig = plt.figure(figsize=(15,9))
        
        # closing price plot 
        ax = fig.add_subplot(2,1,1)
        ax.plot(data["close"], label="Close Price")
        ax.set_ylabel("USDT")
        ax.legend(loc="best")
        ax.grid()

        # positions plot 
        ax = fig.add_subplot(2,1,2)
        ax.plot(data["position"], label="Trading position")
        ax.set_ylabel("Trading Position")
        ax.set_ylim([-1.5, 1.5])

        # show plot 
        plt.show()

    def plot_perf(self, data):

        # equity curve plot
        data["cum_pl"].plot(label="Equity Curve", figsize=(15,8))
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.show()

    def stats_perf(self, data):
        
        # print date from and to 
        days_difference = data.tail(1)["close_time"].values[0] - data.head(1)["open_time"].values[0]
        days_difference = int(round(days_difference / np.timedelta64(1, 'D')))

        time_from = data.head(1)["open_time"].astype(str).values[0]
        time_to = data.tail(1)["close_time"].astype(str).values[0]
        print("From {} to {} ({} days)\n".format(time_from, time_to, days_difference))
        
        # print total number of trades
        # we can sum up every time a fee was applied to get total trades 
        print("Total number of trades: {}".format(data.apply_fee.sum()))

        # print avg. trades per date
        print("Avg. trades per day: {}".format(round(data.apply_fee.sum() / days_difference, 2)))

        # print profit/loss (log returns)
        cum_return = round(data["cum_pl"].iloc[-1] * 100, 2)
        print("Profit/Loss [Log Return]: {0}%".format(cum_return))  

        # # print profit/loss (simple return)
        # simple_return = (np.exp(data.iloc[-1].cum_pl) - 1) * 100
        # print("Profit/Loss [Simple Return]: {0}%".format(round(simple_return, 2)))  

        # print maximum gains (log returns)
        max_cum_pl = round(data["cum_pl"].max() * 100, 2)
        print("Maximum Gain: {0}%".format(max_cum_pl))

        #print maximum loss (log returns) 
        min_cum_pl = round(data["cum_pl"].min() * 100, 2)
        print("Maximum Drawdown: {0}%".format(min_cum_pl))

        # print sharpe ratio 
        # 96 (15 minutes in a day) and 365 days for the crypto market 
        # we compute the sharpe ratio based on profit and loss 
        sharpe_ratio = fs.sharpe_ratio(96*365, data.pl)
        print("Annualised Sharpe Ratio: {0}".format(round(sharpe_ratio, 6)))
