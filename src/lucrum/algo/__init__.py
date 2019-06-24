#from .emacrossover import (EMACrossoverAlgo)
from .macrossover import (MACrossoverAlgo) # moving average crossover strategy 
from .rsisimple import (SimpleRsiAlgo)     # simple rsi strategy
from .buynhold import (BuyHoldStrategy)    # buy and hold strategy 
from .williamrsimple import(SimpleWilliamRAlgo) # william R strategy 
from .natr import (RsiNatrAlgo)            # strategies with a volatility measure 
from .finstats import(dist_moments, sharpe_ratio)  # finance functions 
