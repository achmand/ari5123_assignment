#from .emacrossover import (EMACrossoverAlgo)
from .macrossover import (MACrossoverAlgo) # moving average crossover strategy 
from .rsisimple import (SimpleRsiAlgo)     # simple rsi strategy
from .buynhold import (BuyHoldStrategy)    # buy and hold strategy 
from .mlalgo import (XgbBoostAlgo)         # ml/ai strategies 
from .williamrsimple import(SimpleWilliamRAlgo) # william R strategy 
from .combinedta import(CombinedTaAlgo)    # combination of all indicators
from .natr import (RsiNatrAlgo)            # strategies with a volatility measure 
from .finstats import(dist_moments, sharpe_ratio)  # finance functions 
