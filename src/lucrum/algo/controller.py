"""Base class inherited by trading strategies/algorithms."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from abc import ABC, abstractmethod

###### base trading algorithm class #######################################
class _Controller():   
    
    @abstractmethod
    def gen_features(self, data, *parameters):
        """An abstract method used to generate features for this strategy.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices for a specific asset, expected to have OHLC.

        parameters : dictionary
            Argument list used in the child class to generate features. 
        """

        raise NotImplementedError("Should implement gen_features().")

    @abstractmethod
    def gen_positions(self, data):
        """An abstract method used to generate positions for this strategy.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds data/prices/features for a specific asset.
        """

        raise NotImplementedError("Should implement gen_positions().")

    @abstractmethod
    def evaluate(self, data, trading_fee):
        """Evaluates the performance for this strategy.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds positions for a specific asset.
        trading_fee: float
            The trading fee applied which each trade. 
        """
        
        raise NotImplementedError("Should implement evaluate().")

    @abstractmethod
    def stats_perf(self, data):
        """Prints stats for evaluation for this strategy.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds profit/losses for a specific asset.
        """
        
        raise NotImplementedError("Should implement stats_perf().")

