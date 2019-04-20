"""Script which holds utility functions."""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

def lag_col(df, lag, col, loc_offset=None):
    lagged_cols = []
    for i in range(1, lag + 1):
        lagged = df[col].shift(i)
        lag_col = col + '_lag_' + str(i)
        lagged_cols.append(lag_col)
        tmp_loc = df.shape[1] if loc_offset is None else df.shape[1] - loc_offset
        df.insert(tmp_loc, lag_col, lagged)
    
    return lagged_cols

