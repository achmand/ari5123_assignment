"""A collection of ML/AI strategies. Includes: XgbBoostAlgo, Random Forest and Logistic Regression."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

# TODO -> This copy and paste code with very similar code is getting out of hand 
#  should be written in the abstract controller class (re-usability)
# maybe write another parent class for the ai/ml models

###### importing dependencies #############################################
import numpy as np
import pandas as pd
import xgboost as xgb
import lucrum.algo.pyta as ta 
import lucrum.algo._utils as ut
import matplotlib.pyplot as plt
import lucrum.dataconst as dcons
import lucrum.algo.finstats as fs
from .controller import _Controller
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

###### XGBoost classifier strategy class ##################################
class XgbBoostAlgo(_Controller):
    """A ML strategy which uses XGBoost. Uses a combination of all the other indicators as features. 
       Trains a model and apply a classification which can be considered as a position (up or down).
       A confidence level is also considered before taking a position.       
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
                     lagged_features,
                     create_y= False, 
                     price=dcons.CLOSE):     
        """Generates features for technical indicators.
           Which include EMA, SMA, RSI, William%R and NATR. 
        
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
        willr_window: integer
            The time window used for William%R.
        natr_window: integer
            The time window used for NATR. 
        lagged_features: integer
            Lagged features to be considered. 
        create_y: boolean
            If true creates the y variable used for training. 
        price: str, optional(defaulf='close')
            The column name used to apply the lead and lag moving average on. 
            By default this will be applied to the closing price ('close') 
            but this can be applied to any column with numeric values.  
            
        Returns
        -------
            list of str:
                A list of string containing the feature names. 
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
        
        # create a list for feature names 
        feature_names = []
        
        # apply lagged features to RSI, William%R and NATR
        
        # lagged features for RSI + normalise + append to features list
        lagged_rsi_cols = ut.lag_col(data, lagged_features, "rsi")   
        lagged_rsi_cols += ["rsi"]
        data[lagged_rsi_cols] = data[lagged_rsi_cols].div(data[lagged_rsi_cols].sum(axis=1), axis=0)
        feature_names += lagged_rsi_cols  
        
        # lagged features for William%R + normalise + append to features list
        lagged_willr_cols = ut.lag_col(data, lagged_features, "willr")   
        lagged_willr_cols += ["willr"]
        data[lagged_willr_cols] = data[lagged_willr_cols].div(data[lagged_willr_cols].sum(axis=1), axis=0)
        feature_names += lagged_willr_cols  
        
        # lagged features for NATR + normalise + append to features list
        lagged_natr_cols = ut.lag_col(data, lagged_features, "natr")   
        lagged_natr_cols += ["natr"]
        data[lagged_natr_cols] = data[lagged_natr_cols].div(data[lagged_natr_cols].sum(axis=1), axis=0)
        feature_names += lagged_natr_cols  
        
        # for moving averages first we create lagged for both EMA and SMA 
        
        # create lagged for ema (LEAD) and normalise
        lagged_lead_ema = ut.lag_col(data, lagged_features, "lead_ema")
        lagged_lead_ema += ["lead_ema"]
        data[lagged_lead_ema] = data[lagged_lead_ema].div(data[lagged_lead_ema].sum(axis=1), axis=0)
        
        # create lagged for ema (LAG) and normalise
        lagged_lag_ema = ut.lag_col(data, lagged_features, "lag_ema")
        lagged_lag_ema += ["lag_ema"]
        data[lagged_lag_ema] = data[lagged_lag_ema].div(data[lagged_lag_ema].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged ema (eg. lag_ema - lead_ema, lag_ema_1 - lead_ema_1)
        data["ema_0"] = data["lag_ema"] - data["lead_ema"]
        feature_names += ["ema_0"]
        
        # apply feature reduction on lagged ema 
        for i in range(lagged_features):
            tmp = i + 1
            data["ema_{}".format(tmp)] = data["lag_ema_lag_{}".format(tmp)] - data["lead_ema_lag_{}".format(tmp)] 
            feature_names += ["ema_{}".format(tmp)]
        
        # create lagged for sma (LEAD) and normalise 
        lagged_lead_sma = ut.lag_col(data, lagged_features, "lead_sma")
        lagged_lead_sma += ["lead_sma"]
        data[lagged_lead_sma] = data[lagged_lead_sma].div(data[lagged_lead_sma].sum(axis=1), axis=0)
        
        # create lagged for sma (LAG) and normalise
        lagged_lag_sma = ut.lag_col(data, lagged_features, "lag_sma")
        lagged_lag_sma += ["lag_sma"]
        data[lagged_lag_sma] = data[lagged_lag_sma].div(data[lagged_lag_sma].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged sma (eg. lag_sma - lead_sma, lag_sma_1 - lead_sma_1)
        data["sma_0"] = data["lag_sma"] - data["lead_sma"]
        feature_names += ["sma_0"]
        
        # apply feature reduction on lagged sma 
        for i in range(lagged_features):
            tmp = i + 1
            data["sma_{}".format(tmp)] = data["lag_sma_lag_{}".format(tmp)] - data["lead_sma_lag_{}".format(tmp)] 
            feature_names += ["sma_{}".format(tmp)]
                
        # drop nan values 
        data.dropna(inplace=True)

        # if create_y is true we need to create the y variable 
        # this will be used in training 
        if create_y == True:
            # predict direction
            data["y"] = (data["close"] < data["close"].shift(-1)).astype(int)
        
        return feature_names
    
    def train_algo(self, data, features, max_depth, random_state):
        
        # train XGBoost Classifier 
        return xgb.XGBClassifier(random_state=random_state, 
                                n_jobs=-1, 
                                n_estimators=100, 
                                max_depth=max_depth).fit(data[features], data["y"])
        
        #print("TRAINING COMPLETE")
        
    def show_feature_importance(self, features, clf):
        
        # show feature importance
        feats = {} # a dict to hold feature_name: feature_importance
        for feature, importance in zip(features, clf.feature_importances_):
            feats[feature] = importance #add the name/value pair 

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
        importances.sort_values(by='Importance').plot(kind='bar', rot=45, figsize=(15,9))
    
    def gen_positions(self, data, clf, features, confidence):
        
        # first we generate the classification from our trained model 
        classification = clf.predict(data[features]) # generate classification
        classification_probs = clf.predict_proba(data[features]) # generate classification probs
        probability = np.amax(classification_probs, axis=1) # get prob for classification
        
        # add column for classification and probability 
        data["classification"] = classification
        data["probability"] = probability
        
        # generates signals 
        # long signal (predicted up move and probability is above confidence level)
        data["long_signal"] = ((data["classification"] == 1) & (data["probability"] > confidence)) 
        
        # short signal (predicted down move and probability is above confidence level)
        data["short_signal"] = ((data["classification"] == 0) & (data["probability"] > confidence)) 

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
        
        # if a you take a position and you close it immediately apply 
        # the fee *2 which means you entered and exit at the same candle 
        # eg. 0 1 0 or -1 1 -1 or 1 -1 1
        data.loc[(data.position != 0) & (data.position.shift(1) != data.position ) & (data.position.shift(-1) != data.position) , "apply_fee"] = 2 

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
        
        # calculate profit and loss + tx cost 
        data["pl"] = (data["position"].shift(1) * data["log_returns"]) - ((data.apply_fee.shift(1) * trading_fee) * np.abs(data["log_returns"]))

        # cumulative profit and loss 
        data["cum_pl"] = data["pl"].cumsum()
    
    def evaluate_classifier(self, clf, x, true_y):
        predicted_y = clf.predict(x)
        print("Classifier Accuracy: {}".format(accuracy_score(true_y, predicted_y)))
        print("Classifier F1 Score: {}".format(f1_score(true_y, predicted_y, average="binary")))
        
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
        total_trades = data.apply_fee.sum()
        print("Total number of trades: {}".format(total_trades))

        # print avg. trades per date
        print("Avg. trades per day: {}".format(round(total_trades / days_difference, 2)))

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

###### Random forest classifier strategy class ############################
class RandomForestAlgo(_Controller):
    """A ML strategy which uses Random Forest. Uses a combination of all the other indicators as features. 
       Trains a model and apply a classification which can be considered as a position (up or down).
       A confidence level is also considered before taking a position.       
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
                     lagged_features,
                     create_y= False, 
                     price=dcons.CLOSE):     
        """Generates features for technical indicators.
           Which include EMA, SMA, RSI, William%R and NATR. 
        
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
        willr_window: integer
            The time window used for William%R.
        natr_window: integer
            The time window used for NATR. 
        lagged_features: integer
            Lagged features to be considered. 
        create_y: boolean
            If true creates the y variable used for training. 
        price: str, optional(defaulf='close')
            The column name used to apply the lead and lag moving average on. 
            By default this will be applied to the closing price ('close') 
            but this can be applied to any column with numeric values.  
            
        Returns
        -------
            list of str:
                A list of string containing the feature names. 
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
        
        # create a list for feature names 
        feature_names = []
        
        # apply lagged features to RSI, William%R and NATR
        
        # lagged features for RSI + normalise + append to features list
        lagged_rsi_cols = ut.lag_col(data, lagged_features, "rsi")   
        lagged_rsi_cols += ["rsi"]
        data[lagged_rsi_cols] = data[lagged_rsi_cols].div(data[lagged_rsi_cols].sum(axis=1), axis=0)
        feature_names += lagged_rsi_cols  
        
        # lagged features for William%R + normalise + append to features list
        lagged_willr_cols = ut.lag_col(data, lagged_features, "willr")   
        lagged_willr_cols += ["willr"]
        data[lagged_willr_cols] = data[lagged_willr_cols].div(data[lagged_willr_cols].sum(axis=1), axis=0)
        feature_names += lagged_willr_cols  
        
        # lagged features for NATR + normalise + append to features list
        lagged_natr_cols = ut.lag_col(data, lagged_features, "natr")   
        lagged_natr_cols += ["natr"]
        data[lagged_natr_cols] = data[lagged_natr_cols].div(data[lagged_natr_cols].sum(axis=1), axis=0)
        feature_names += lagged_natr_cols  
        
        # for moving averages first we create lagged for both EMA and SMA 
        
        # create lagged for ema (LEAD) and normalise
        lagged_lead_ema = ut.lag_col(data, lagged_features, "lead_ema")
        lagged_lead_ema += ["lead_ema"]
        data[lagged_lead_ema] = data[lagged_lead_ema].div(data[lagged_lead_ema].sum(axis=1), axis=0)
        
        # create lagged for ema (LAG) and normalise
        lagged_lag_ema = ut.lag_col(data, lagged_features, "lag_ema")
        lagged_lag_ema += ["lag_ema"]
        data[lagged_lag_ema] = data[lagged_lag_ema].div(data[lagged_lag_ema].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged ema (eg. lag_ema - lead_ema, lag_ema_1 - lead_ema_1)
        data["ema_0"] = data["lag_ema"] - data["lead_ema"]
        feature_names += ["ema_0"]
        
        # apply feature reduction on lagged ema 
        for i in range(lagged_features):
            tmp = i + 1
            data["ema_{}".format(tmp)] = data["lag_ema_lag_{}".format(tmp)] - data["lead_ema_lag_{}".format(tmp)] 
            feature_names += ["ema_{}".format(tmp)]
        
        # create lagged for sma (LEAD) and normalise 
        lagged_lead_sma = ut.lag_col(data, lagged_features, "lead_sma")
        lagged_lead_sma += ["lead_sma"]
        data[lagged_lead_sma] = data[lagged_lead_sma].div(data[lagged_lead_sma].sum(axis=1), axis=0)
        
        # create lagged for sma (LAG) and normalise
        lagged_lag_sma = ut.lag_col(data, lagged_features, "lag_sma")
        lagged_lag_sma += ["lag_sma"]
        data[lagged_lag_sma] = data[lagged_lag_sma].div(data[lagged_lag_sma].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged sma (eg. lag_sma - lead_sma, lag_sma_1 - lead_sma_1)
        data["sma_0"] = data["lag_sma"] - data["lead_sma"]
        feature_names += ["sma_0"]
        
        # apply feature reduction on lagged sma 
        for i in range(lagged_features):
            tmp = i + 1
            data["sma_{}".format(tmp)] = data["lag_sma_lag_{}".format(tmp)] - data["lead_sma_lag_{}".format(tmp)] 
            feature_names += ["sma_{}".format(tmp)]
                
        # drop nan values 
        data.dropna(inplace=True)

        # if create_y is true we need to create the y variable 
        # this will be used in training 
        if create_y == True:
            # predict direction
            data["y"] = (data["close"] < data["close"].shift(-1)).astype(int)
        
        return feature_names
    
    def train_algo(self, data, features, max_depth, random_state):
        
        # train Random Forest Classifier 
        return RandomForestClassifier(random_state=random_state, 
                                      n_jobs=-1, 
                                      n_estimators=100, 
                                      max_depth=max_depth).fit(data[features], data["y"])
        
        #print("TRAINING COMPLETE")
        
    def show_feature_importance(self, features, clf):
        
        # show feature importance
        feats = {} # a dict to hold feature_name: feature_importance
        for feature, importance in zip(features, clf.feature_importances_):
            feats[feature] = importance #add the name/value pair 

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
        importances.sort_values(by='Importance').plot(kind='bar', rot=45, figsize=(15,9))
    
    def gen_positions(self, data, clf, features, confidence):
        
        # first we generate the classification from our trained model 
        classification = clf.predict(data[features]) # generate classification
        classification_probs = clf.predict_proba(data[features]) # generate classification probs
        probability = np.amax(classification_probs, axis=1) # get prob for classification
        
        # add column for classification and probability 
        data["classification"] = classification
        data["probability"] = probability
        
        # generates signals 
        # long signal (predicted up move and probability is above confidence level)
        data["long_signal"] = ((data["classification"] == 1) & (data["probability"] > confidence)) 
        
        # short signal (predicted down move and probability is above confidence level)
        data["short_signal"] = ((data["classification"] == 0) & (data["probability"] > confidence)) 

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
        
        # if a you take a position and you close it immediately apply 
        # the fee *2 which means you entered and exit at the same candle 
        # eg. 0 1 0 or -1 1 -1 or 1 -1 1
        data.loc[(data.position != 0) & (data.position.shift(1) != data.position ) & (data.position.shift(-1) != data.position) , "apply_fee"] = 2 

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
               
        # calculate profit and loss + tx cost 
        data["pl"] = (data["position"].shift(1) * data["log_returns"]) - ((data.apply_fee.shift(1) * trading_fee) * np.abs(data["log_returns"]))

        # cumulative profit and loss 
        data["cum_pl"] = data["pl"].cumsum()
    
    def evaluate_classifier(self, clf, x, true_y):
        predicted_y = clf.predict(x)
        print("Classifier Accuracy: {}".format(accuracy_score(true_y, predicted_y)))
        print("Classifier F1 Score: {}".format(f1_score(true_y, predicted_y, average="binary")))
        
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
        total_trades = data.apply_fee.sum()
        print("Total number of trades: {}".format(total_trades))

        # print avg. trades per date
        print("Avg. trades per day: {}".format(round(total_trades / days_difference, 2)))

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

###### Logistic Regression classifier strategy class ######################
class LogRegAlgo(_Controller):
    """A ML strategy which uses Logistic Regression. Uses a combination of all the other indicators as features. 
       Trains a model and apply a classification which can be considered as a position (up or down).
       A confidence level is also considered before taking a position.       
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
                     lagged_features,
                     create_y= False, 
                     price=dcons.CLOSE):     
        """Generates features for technical indicators.
           Which include EMA, SMA, RSI, William%R and NATR. 
        
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
        willr_window: integer
            The time window used for William%R.
        natr_window: integer
            The time window used for NATR. 
        lagged_features: integer
            Lagged features to be considered. 
        create_y: boolean
            If true creates the y variable used for training. 
        price: str, optional(defaulf='close')
            The column name used to apply the lead and lag moving average on. 
            By default this will be applied to the closing price ('close') 
            but this can be applied to any column with numeric values.  
            
        Returns
        -------
            list of str:
                A list of string containing the feature names. 
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
        
        # create a list for feature names 
        feature_names = []
        
        # apply lagged features to RSI, William%R and NATR
        
        # lagged features for RSI + normalise + append to features list
        lagged_rsi_cols = ut.lag_col(data, lagged_features, "rsi")   
        lagged_rsi_cols += ["rsi"]
        data[lagged_rsi_cols] = data[lagged_rsi_cols].div(data[lagged_rsi_cols].sum(axis=1), axis=0)
        feature_names += lagged_rsi_cols  
        
        # lagged features for William%R + normalise + append to features list
        lagged_willr_cols = ut.lag_col(data, lagged_features, "willr")   
        lagged_willr_cols += ["willr"]
        data[lagged_willr_cols] = data[lagged_willr_cols].div(data[lagged_willr_cols].sum(axis=1), axis=0)
        feature_names += lagged_willr_cols  
        
        # lagged features for NATR + normalise + append to features list
        lagged_natr_cols = ut.lag_col(data, lagged_features, "natr")   
        lagged_natr_cols += ["natr"]
        data[lagged_natr_cols] = data[lagged_natr_cols].div(data[lagged_natr_cols].sum(axis=1), axis=0)
        feature_names += lagged_natr_cols  
        
        # for moving averages first we create lagged for both EMA and SMA 
        
        # create lagged for ema (LEAD) and normalise
        lagged_lead_ema = ut.lag_col(data, lagged_features, "lead_ema")
        lagged_lead_ema += ["lead_ema"]
        data[lagged_lead_ema] = data[lagged_lead_ema].div(data[lagged_lead_ema].sum(axis=1), axis=0)
        
        # create lagged for ema (LAG) and normalise
        lagged_lag_ema = ut.lag_col(data, lagged_features, "lag_ema")
        lagged_lag_ema += ["lag_ema"]
        data[lagged_lag_ema] = data[lagged_lag_ema].div(data[lagged_lag_ema].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged ema (eg. lag_ema - lead_ema, lag_ema_1 - lead_ema_1)
        data["ema_0"] = data["lag_ema"] - data["lead_ema"]
        feature_names += ["ema_0"]
        
        # apply feature reduction on lagged ema 
        for i in range(lagged_features):
            tmp = i + 1
            data["ema_{}".format(tmp)] = data["lag_ema_lag_{}".format(tmp)] - data["lead_ema_lag_{}".format(tmp)] 
            feature_names += ["ema_{}".format(tmp)]
        
        # create lagged for sma (LEAD) and normalise 
        lagged_lead_sma = ut.lag_col(data, lagged_features, "lead_sma")
        lagged_lead_sma += ["lead_sma"]
        data[lagged_lead_sma] = data[lagged_lead_sma].div(data[lagged_lead_sma].sum(axis=1), axis=0)
        
        # create lagged for sma (LAG) and normalise
        lagged_lag_sma = ut.lag_col(data, lagged_features, "lag_sma")
        lagged_lag_sma += ["lag_sma"]
        data[lagged_lag_sma] = data[lagged_lag_sma].div(data[lagged_lag_sma].sum(axis=1), axis=0)
        
        # apply feature reduction of these two 
        # since we are interested in the value with respect to each other 
        # we subtract the lag with the lead for all lagged sma (eg. lag_sma - lead_sma, lag_sma_1 - lead_sma_1)
        data["sma_0"] = data["lag_sma"] - data["lead_sma"]
        feature_names += ["sma_0"]
        
        # apply feature reduction on lagged sma 
        for i in range(lagged_features):
            tmp = i + 1
            data["sma_{}".format(tmp)] = data["lag_sma_lag_{}".format(tmp)] - data["lead_sma_lag_{}".format(tmp)] 
            feature_names += ["sma_{}".format(tmp)]
                
        # drop nan values 
        data.dropna(inplace=True)

        # if create_y is true we need to create the y variable 
        # this will be used in training 
        if create_y == True:
            # predict direction
            data["y"] = (data["close"] < data["close"].shift(-1)).astype(int)
        
        return feature_names
    
    def train_algo(self, data, features, random_state):
        
        # train LogReg Classifier 
        return LogisticRegression(penalty="l2",
                                  solver="lbfgs",
                                  max_iter=1000,
                                  n_jobs=-1).fit(data[features], data["y"])

        #print("TRAINING COMPLETE")
        
    def show_feature_importance(self, features, clf):
        feats = {}
        i = 0
        for feat in features:
            feats[feat] = clf.coef_[0][i]
            i = i+1 
        
        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Coefficients'})
        importances.sort_values(by='Coefficients').plot(kind='bar', rot=45, figsize=(15,9))
    
    def gen_positions(self, data, clf, features, confidence):
        
        # first we generate the classification from our trained model 
        classification = clf.predict(data[features]) # generate classification
        classification_probs = clf.predict_proba(data[features]) # generate classification probs
        probability = np.amax(classification_probs, axis=1) # get prob for classification
        
        # add column for classification and probability 
        data["classification"] = classification
        data["probability"] = probability
        
        # generates signals 
        # long signal (predicted up move and probability is above confidence level)
        data["long_signal"] = ((data["classification"] == 1) & (data["probability"] > confidence)) 
        
        # short signal (predicted down move and probability is above confidence level)
        data["short_signal"] = ((data["classification"] == 0) & (data["probability"] > confidence)) 

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
        
        # if a you take a position and you close it immediately apply 
        # the fee *2 which means you entered and exit at the same candle 
        # eg. 0 1 0 or -1 1 -1 or 1 -1 1
        data.loc[(data.position != 0) & (data.position.shift(1) != data.position ) & (data.position.shift(-1) != data.position) , "apply_fee"] = 2 
  
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
        
        # calculate profit and loss + tx cost 
        data["pl"] = (data["position"].shift(1) * data["log_returns"]) - ((data.apply_fee.shift(1) * trading_fee) * np.abs(data["log_returns"]))

        # cumulative profit and loss 
        data["cum_pl"] = data["pl"].cumsum()
    
    def evaluate_classifier(self, clf, x, true_y):
        predicted_y = clf.predict(x)
        print("Classifier Accuracy: {}".format(accuracy_score(true_y, predicted_y)))
        print("Classifier F1 Score: {}".format(f1_score(true_y, predicted_y, average="binary")))
        
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
        total_trades = data.apply_fee.sum()
        print("Total number of trades: {}".format(total_trades))

        # print avg. trades per date
        print("Avg. trades per day: {}".format(round(total_trades / days_difference, 2)))

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
