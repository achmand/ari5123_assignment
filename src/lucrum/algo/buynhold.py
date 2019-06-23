"""Buy and hold strategy/algorithm."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import matplotlib.pyplot as plt
import lucrum.dataconst as dcons
from .controller import _Controller

###### buy n hold strategy class ##########################################
class BuyHoldStrategy(_Controller):
    """Simple baseline model which uses the buy and hold strategy.      
    """  

    # for this strategy there is no need to generate features 
    def gen_features(self, data, *parameters):
        raise NotImplementedError("Should implement gen_features().")

    # for this strategy there is no need to generate positions 
    def gen_positions(self, data):
        """Generates postions this strategy is always in a hold position (bought once).

        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices/features for a specific asset.
        """

        # always in a position (Holding)
        data["position"] = 1

    def evaluate(self, data, trading_fee):
        """Evaluates/calculates performance for the buy n hold strategy.  

        Parameters 
        ----------
        data: pandas dataframe 
            Holds the prices for the asset being evaluated.
        trading_fee: float
            The trading fee applied which each trade. 
        """

        # firstly calculate log returns
        data["logprices"] = np.log(data[dcons.CLOSE])
        data["log_returns"] = data.logprices - data.logprices.shift(1)

        # since we execute only one trade apply trading fee to first log return only 
        data.iloc[1, data.columns.get_loc("log_returns")] = (1-trading_fee) * data.iloc[1]["log_returns"]

        # to get p/l we just need to sum up the log returns 
        data["cum_pl"] = data["log_returns"].cumsum()

    def plot_pos(self, data):
        fig = plt.figure(figsize=(15,9))
        ax = fig.add_subplot(2,1,1)

        ax.plot(data["close"], label="Close Price")

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
        time_from = data.head(1)["open_time"].astype(str).values[0]
        time_to = data.tail(1)["close_time"].astype(str).values[0]
        print("From {} to {}".format(time_from, time_to))
        
        # print price bought at and current price 
        bought_price = data.iloc[0]["close"]
        current_price = data.iloc[-1]["close"]
        print("Bought at price {} and current price is {}\n".format(round(bought_price, 5), round(current_price, 5)))

        # print total number of trades, in this strategy always 1 
        print("Total number of trades: {}".format(1))

        # print std. of return per trade in this case N/A as one trade was executed 
        print("Standard Deviation of return per trade: {}".format("N/A"))

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
        # since we are always holding use log returns 
        sharpe_ratio = np.sqrt(96 * 365) * data.log_returns.mean() / data.log_returns.std()
        print("Annualised Sharpe Ratio: {0}".format(round(sharpe_ratio, 6)))

###########################################################################

