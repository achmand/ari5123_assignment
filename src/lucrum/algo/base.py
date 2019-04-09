"""Base class inherited by other algo controllers."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################

###### base algo controller class #########################################
class _Controller():

    
    def gen_signals(self, data, parameters):
        pass

    def hey(self):
        print("S")