# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

# TODO -> Should I create an interface for signals generated using Machine Learning ??

###### importing dependencies #############################################
from lucrum.algo import pyta
import lucrum.datareader as ldr
from lucrum.algo import _utils as utl
import lucrum.datareader.dataconst as dcons

###### stacked (ensemble) classifier class ################################

###### moving average crossover machine learning class ####################
class MaCrssoverMl():
    """Predicts moving average crossover using machine learning.      
    """

    def __init__(self, lead_ma, lead_time, lag_ma, lag_time, price=dcons.CLOSE):        
        """Moving average prediction constructor. 

        Parameters 
        ----------
        
        Attributes
        ----------
        
        References
        ----------
        """

        self.lead_ma = lead_ma
        self.lead_time = lead_time
        self.lag_ma = lag_ma
        self.lag_time = lag_time 
        self.price = price 
        
    @property
    def data(self):
        return self._data

    @property
    def instances(self):
        return self._df
    
    def get_data(self, source, *args, **kwargs):
        # gets data from the source specified 
        self._data = ldr.get_data(source, *args, **kwargs)

    def set_data(self, df):
        # preprocess & set data from the df passed 
        self._data = df

    def _pre_process(self, df):

        # lead/lag df columns in the df 
        self._lead_col = self.lead_ma + "_" + str(self.lead_time)
        self._lag_col = self.lag_ma + "_" + str(self.lag_time)

        # set config for moving avgs indicators 
        ta_config = {
            self.lead_ma:[(self._lead_col, {"timeperiod":self.lead_time, "price":self.price})],
            self.lag_ma:[(self._lag_col, {"timeperiod":self.lag_time, "price":self.price})]
        }

        # apply moving avg indicators 
        pyta.apply_ta(df, ta_config)

        # get reference for lead and lag ma
        tmp_lead = df[self._lead_col].shift(1)
        tmp_lag = df[self._lag_col].shift(1)
        
        # compute outcome (crossover)
        df["crossover"] = (((df[self._lead_col] < df[self._lag_col]) & (tmp_lead >= tmp_lag))
            | ((df[self._lead_col] > df[self._lag_col]) & (tmp_lead <= tmp_lag)))

        # change crossover to binary 
        df["crossover"] = df["crossover"].map({True: 1, False: 0})
        
        # drop na from df 
        df = df.dropna()
        return df

    def _gen_features(self, lag, distance, ta_config=None):

        # preprocess data for ma crossovers
        df = self._pre_process(self._data.copy())

        # create lagged MA lead & lag 
        lagged_cols = utl.lag_col(df, lag, self._lead_col, 1)
        lagged_cols += utl.lag_col(df, lag, self._lag_col, 1)

        # drop na from df
        df = df.dropna()
        feature_list = []

        # create features from lagged MA 
        # 1. normalize lagged MA row level (lead & lag) 
        self._df = df
        self._df[lagged_cols] = df[lagged_cols].div(df[lagged_cols].sum(axis=1), axis=0)

        # 2. feature reduction, subtracts lag with lead, row by row 
        for i in range(0, lag + 1):
            ma_col = "ma_" + str(i)
            feature_list.append(ma_col)

            tmp_lag_col = self._lag_col if i==0 else self._lag_col + "_lag_" + str(i)
            tmp_lead_col = self._lead_col if i==0 else self._lead_col + "_lag_" + str(i)

            ma_diff = self._df[tmp_lag_col] - self._df[tmp_lead_col]
            self._df.insert(self._df.shape[1] - 1, ma_col, ma_diff)
        
        # remove the lagged columns now since features were extracted
        self._df.drop(lagged_cols, axis=1, inplace=True)

        # shift crossover since we want to train model to classify crossover
        crossover = self._df["crossover"].shift(-distance).fillna(-1).astype(int).copy()
        self._df["crossover"] = crossover
        self._df.drop(self._df.tail(distance).index, inplace=True)

        return feature_list

###### references #########################################################
"""[1] Machine learning applied to financial markets: moving average crossover prediction 
    => http://www.liverium.com/trading/machine-learning-applied-to-financial-markets-moving-average-crossover-prediction/ 
"""
############################################################################