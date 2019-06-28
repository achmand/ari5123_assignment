# Genetic Algorithm and Machine Learning Signals for the Cryptocurrency Market

This is my final project for the ‘Intelligent Algorithmic Trading Assignment’ for the ARI5123 study unit. The final results for this project is best viewed using nbviewer, click on the following link to view results [Jupyter Notebook](https://nbviewer.jupyter.org/github/achmand/ari5123_assignment/blob/master/src/algo_trading.ipynb?flush_cache=true).

Resources for this project;

* [algo module](https://github.com/achmand/ari5123_assignment/tree/master/src/lucrum/algo) [Holds different trading strategies including AI/ML strategies]
* [datareader module](https://github.com/achmand/ari5123_assignment/tree/master/src/lucrum/datareader) [Reads data from Binance, also have abstractions to be able to add other data sources]
* [Jupyter Notebook/Results](https://github.com/achmand/ari5123_assignment/blob/master/src/algo_trading.ipynb) [Notebook with visualisations and results]
* [Datasets](https://github.com/achmand/ari5123_assignment/tree/master/src/data) [Datasets collected]
* [Anaconda Environment](https://github.com/achmand/ari5123_assignment/blob/master/src/environment.yml) 
* [Research Paper](https://github.com/achmand/ari5123_assignment/blob/master/doc/intelligent_algo_trading.pdf) 

## Setup
### Environment
Python version 3.7.3
Running under: Ubuntu 18.04.1 LTS
IDE/Text Editor: [Visual Code](https://code.visualstudio.com/) 

 An anaconda environment file ([environment.yml](https://github.com/achmand/ari5123_assignment/blob/master/src/environment.yml)) is supplied to be able to run the Jupyter Notebook. If you don’t have anaconda installed on your system, follow this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart) to install anaconda on Ubuntu 18.04. Once anaconda is set up on your system execute the following commands to create an new environment from the supplied yaml file.

```
conda env create --f environment.yaml # creates a conda environment from file
conda activate ari5123 # activate conda environment 
jupyter notebook # run jupyter notebook
```

## Scope for this project 

Cryptocurrencies have attracted a lot of attention in recent years and the use of AI/ML models to assist or perform automated trading is on the rise. In this study, we use the approach of training three different classification models in particular XGBoost, Random Forest and Logistic Regression using technical indicators as features to test the inefficiencies of the cryptocurrency market to generate risk-adjusted returns. Furthermore, we try to combine these models into one trading strategy using a Genetic Algorithm. We analysed 15 minute data for the following pairs: BTC/USDT, ETH/USDT, XRP/USDT, LTC/USDT, EOS/USDT and XLM/USDT. Our results show that, AI/ML models can outperform simple
strategies in particular the buy and hold strategy.

The main objectives of this study are as follows:
* Use technical indicators as features to train ML classifiers to predict an up or down move in the price (XGBoost, Random Forest and Logistic Regression)
* Evaluate the classifiers in terms of Accuracy and F1 score
* Apply a meta-heuristic optimisation (optimise on Sharpe Ratio) to find optimal trading rules when combining signals from the trained classifiers (Genetic Algorithm)
* Asses performance in terms of P/L and Annualised Sharpe ratio by running a trading simulation for all the proposed models by taking trading positions based on the predicted outcome
* Compare the proposed models against various other baseline models such as the buy and hold and standard technical indicator strategies (same indicators which were used as features)

The main contributions of our work are:
* Analysing the performance of our proposed models in terms of P/L and Sharpe Ratio and how they compare against standard technical indicator strategies
* Investigating the use of Genetic Algorithm to combine the machine learning models into one trading strategy
* Analysing the performance in terms of P/L and Sharpe Ratio of the combined machine learning models strategy against using the individual models

## Some additional resource 
* [Technical Analysis for Algorithmic Pattern Recognition](https://www.springer.com/gp/book/9783319236353)
* [Awesome Quant Machine Learning Trading Repository](https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading)



