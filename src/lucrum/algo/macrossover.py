"""A moving average (SMA OR EMA) crossover strategy/algorithm."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np 
import lucrum.algo.pyta as ta 
import matplotlib.pyplot as plt
import lucrum.dataconst as dcons
from .controller import _Controller

###### moving average crossover algorithm #################################
class MACrossoverAlgo(_Controller):   
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

        # the same type of moving average 
        if lead == lag:
            ta_config = {
                lead:[("lead_ma", {"timeperiod":lead_t, "price":price}),
                      ("lag_ma", {"timeperiod":lag_t, "price":price})]
            }
        # different types of moving average
        else: 
            ta_config = {
                lead:[("lead_ma", {"timeperiod":lead_t, "price":price})],
                lag:[("lag_ma",   {"timeperiod":lag_t, "price":price})]
            }
        
        # apply moving averages to dataframe
        ta.apply_ta(data, ta_config)

    def gen_positions(self, data):
        """Generates postions based on the features generated (lead/lag moving avgs).
           This generates both short and long positions. 

        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices/features for a specific asset.
        """

        # generates signals 
        data["long_signal"] = data["lead_ma"] > data["lag_ma"]   # long signal
        data["short_signal"] = data["lead_ma"] <= data["lag_ma"] # short signal

        # generate positions 
        data["long_position"] = 0  # initially set to 0 
        data["short_position"] = 0 # initially set to 0
        data.loc[data.long_signal, "long_position"] = 1    # set position to go long
        data.loc[data.short_signal, "short_position"] = -1 # set position to short 

        # set position at time t 
        data["position"] = data.long_position + data.short_position

        # once we generate positions find indices where to apply tx cost 
        # basically we check for any changes between long and short 
        # except when we are not in a trade at all
        data["apply_fee"] = 0
        fee_indices = data[data["position"] != 0]["position"].diff()[data["position"].diff() != 0].index.values
        data.loc[fee_indices, "apply_fee"] = 1 # fee for entry trade 

        # we know that we are always in a trade with this method
        # so since we applied a fee for positions where there is a change from 1 to -1 or 1 to -1
        # we need to add a fee to the index previous to the change (exiting a trade)
        fee_indices = (fee_indices - 1)[1:] # we skip the first instance since it's the first entry 
        data.loc[fee_indices, "apply_fee"] = 1  # apply fee for exiting trades 

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
        fee_indices = fee_indices[fee_indices<data.shape[0]]

        # now we apply fee to the p/l at location where the trade was made + 1
        data.loc[fee_indices , "pl"] =  (1 - trading_fee) * data["pl"]

        # cumulative profit and loss 
        data["cum_pl"] = data["pl"].cumsum()


    def plot_pos(self, data):
        fig = plt.figure(figsize=(15,9))
        ax = fig.add_subplot(2,1,1)

        ax.plot(data["close"], label="Close Price")
        ax.plot(data["lead_ma"], label="Short MA")
        ax.plot(data["lag_ma"], label="Long MA")

        ax.set_ylabel("USDT")
        ax.legend(loc="best")
        ax.grid()

        ax = fig.add_subplot(2,1,2)

        ax.plot(data["position"], label="Trading position")
        ax.set_ylabel("Trading Position")
        plt.show()

    def plot_perf(self, data):
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
        

        # print profit/loss (simple return)
        simple_return = (np.exp(data.iloc[-1].cum_pl) - 1) * 100
        print("Profit/Loss [Simple Return]: {0}%".format(round(simple_return, 2)))  

        # print maximum gains (log returns)
        max_cum_pl = round(data["cum_pl"].max() * 100, 2)
        print("Maximum Gain: {0}%".format(max_cum_pl))

        #print maximum loss (log returns) 
        min_cum_pl = round(data["cum_pl"].min() * 100, 2)
        print("Maximum Drawdown: {0}%".format(min_cum_pl))

        # print sharpe ratio 
        # 96 (15 minutes in a day) and 365 days for the crypto market 
        # we compute the sharpe ratio based on profit and loss 
        sharpe_ratio = np.sqrt(96 * 365) * data.pl.mean() / data.pl.std()
        print("Annualised Sharpe Ratio: {0}".format(round(sharpe_ratio, 6)))