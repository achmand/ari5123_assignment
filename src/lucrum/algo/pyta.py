# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
from talib import abstract
import lucrum.dataconst as dcons

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

def crossover(data, indicator_a, indicator_a_time, indicator_b, indicator_b_time, price=dcons.CLOSE, dropnan=True):
    
    # indicators TA config 
    indicator_a_col = indicator_a + str(indicator_a_time)
    indicator_b_col = indicator_b + str(indicator_b_time)
    ta_config = {
        indicator_a:[(indicator_a_col, {"timeperiod":indicator_a_time, "price":price})],
        indicator_b:[(indicator_b_col, {"timeperiod":indicator_b_time, "price":price})]
    }

    # apply indicators to data
    apply_ta(data, ta_config)

    # drop nan values from indicators if specified
    if dropnan == True:
        data.dropna(inplace=True)

    # get reference for lead and lag ma
    tmp_a = data[indicator_a_col].shift(1)
    tmp_b = data[indicator_b_col].shift(1)
    
    # compute crossover for each instance
    data["crossover"] = (((data[indicator_a_col] < data[indicator_b_col]) & (tmp_a >= tmp_b))
        | ((data[indicator_a_col] > data[indicator_b_col]) & (tmp_a <= tmp_b)))

    # change crossover outcome to binary 
    data["crossover"] = data["crossover"].map({True: 1, False: 0})

def lag_col(data, lag, col, loc_offset=None, dropnan=True):
    lagged_cols = []
    for i in range(1, lag + 1):
        lagged = data[col].shift(i)
        lag_col = col + '_lag_' + str(i)
        lagged_cols.append(lag_col)
        tmp_loc = data.shape[1] if loc_offset is None else data.shape[1] - loc_offset
        data.insert(tmp_loc, lag_col, lagged)
    
    # drop nan values from indicators if specified
    if dropnan == True:
        data.dropna(inplace=True)

    return lagged_cols
    
###########################################################################
