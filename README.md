# MomentumStrat
A Python implementation of a Long only momentum strategy on top decile stocks from NIFTY100 based on the paper:

Raju, Rajan and Chandrasekaran, Abhijit, Implementing a Systematic Long-only Momentum 
Strategy: Evidence From India (October 24, 2019). 
Available at SSRN: https://ssrn.com/abstract=3510433 or http://dx.doi.org/10.2139/ssrn.3510433

NIFTY100 stock data taken from https://www.kaggle.com/kmldas/nse-top-100-stocks

Returns generated from taking long positions rebalanced monthly (based on closing prices, equally weighted) distrubution and time series:

<img src="https://github.com/SpagetBakemono/MomentumStrat/blob/main/Plots/momentumReturns.png" width="400" height="300">

KS Test on the momentum returns: ks statistic =  0.07782137296493385, p value =  0.45508830166356795
