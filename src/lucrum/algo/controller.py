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
    def evaluate(self, data):
        """Evaluates the performance for this strategy.
        
        Parameters 
        ----------
        data: pandas dataframe 
            Holds positions for a specific asset.
        """
        
        raise NotImplementedError("Should implement evaluate().")
