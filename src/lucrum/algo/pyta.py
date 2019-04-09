# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from talib import abstract
import lucrum.datareader.dataconst as dcons

###### technical analysis #################################################
def apply_ta(data, config):
    
    # inputs for abstraction
    ta_inputs = {
        "open": data[dcons.OPEN],
        "high": data[dcons.HIGH],
        "low": data[dcons.LOW],
        "close": data[dcons.CLOSE],
        "volume": data[dcons.VOLUME]
    }

    # loop in every indicator passed
    ta_columns = []
    for key, configs in config.items():
        for config in configs:
            column = config[0]
            config_i = config[1]
            results = abstract.Function(key,ta_inputs, **config_i).outputs
            if isinstance(column, list):
                for i in range(len(results)):
                    data[column[i]] = results[i]
            else:
                data[column] = results
            
            ta_columns.append(column)

    return ta_columns

###########################################################################
