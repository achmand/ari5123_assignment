"""Base class inherited by datareaders to interact with different APIs."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from abc import ABC, abstractmethod

###### base data api class ################################################
class BaseData():
    """Base class for datareaders used to get historical data.      
    """

    def __init__(self, symbols, start, end):
        """Datareader base class constructor.

        Parameters 
        ----------
        symbols : str, List[str]
            List or single symbol used when retrieving data from some datareader. 
        start: str 
            The start date to fetch data from. 
        end: str 
            The end date to fetch data from. 

        Attributes
        ----------
        symbols : str, List[str]
            List or single symbol used when retrieving data from some datareader. 
        start: str 
            The start date to fetch data from. 
        end: str 
            The end date to fetch data from. 
        """

        # set attributes from parameters
        self.symbols = symbols
        self.start = start
        self.end = end 

        # call base constructor 
        super().__init__()

    @abstractmethod
    def close(self):
        """An abstract method used to close sessions/streams in child class."""
        pass

    def read(self):
        """Gets data from datareader."""
        try: 
            # call child read function
            return self.read_datareader()
        finally:
            # call child close function 
            self.close()

    @abstractmethod
    def read_datareader(self):
        """An abstract method used to retrieve the data from the child class."""
        pass 

###########################################################################
