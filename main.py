# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:18:36 2021

@author: Aditya
"""

from saveData import save_data
from pandas.tseries.offsets import MonthEnd
import plots
import pandas as pd
import os
import h5py
import numpy as np
from scipy.stats import kstest

class stocksData:
    """
    class for fetching individual ticker data from HDF5 files
    Generates a dictionary of [ticker]->daily returns
    """
    
    #initialize member variables
    def __init__(self):
        self.stocks = {}
        self.tickers = []
    
    #fetch data from h5 files in /Saved   
    def fetch_data(self, filepath):
        for file in os.listdir(filepath):
            filedir = filepath+"/"+file
            f = h5py.File(filedir, 'r')
            key = list(f.keys())[0]
            self.stocks[key] = pd.read_hdf(path_or_buf=filedir, key=key, mode='r')
        self.tickers = list(self.stocks.keys())
        return self.stocks, self.tickers

class Factor:
    
    class Momentum():
        
        def __init__(self, stocks, tickers):
            self.pricesDict = {}
            self.momentum = {}
            self.prices = pd.DataFrame()
            self.momentums = pd.DataFrame()
            self.stocks = stocks
            self.tickers = tickers
            
        def momentum_return(self, priceType, method="last", freq="1M", period=12, skip=1):
            for ticker in self.stocks:
                df = self.stocks[ticker]
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
                if (method == "last"):
                    prices_df = df.resample(freq).last()
                if (method == "mean"):
                    prices_df = df.resample(freq).mean()
                self.pricesDict[ticker] = prices_df[priceType]
            self.prices = pd.concat((self.pricesDict).values(), axis=1)
            self.prices.columns = self.tickers
            self.momentums = np.log(self.prices.shift(skip)/self.prices.shift(period))
            return self.momentums
        
        def top_n_percentile(self, top_n):
            top_returns = self.momentums.copy()
            i = 0
            for col in top_returns.columns:
                top_returns[col].values[:] = 0
            for date,tickers in self.momentums.iterrows():
                top_performers = tickers.nlargest(top_n).index.values
                for x in top_performers:
                    top_returns[x].iloc[i] = 1
                i = i+1
            return top_returns
        
        def compute_prtfolio_returns(self, one_hot, n_stocks, period):
            self.momentum["P_ret"] = (self.momentums*one_hot).sum(axis=1)/n_stocks
            return self.momentum["P_ret"][period:]
    
class stats_tests():
    
    def ks_test(self, x):
        mean = x.mean()
        std = x.std(ddof=0)
        kstat, pvalue = kstest(x.values, "norm", args=(mean,std),
                               alternative="two-sided", mode='approx')
        print("ks statistic = ", kstat)
        print("p value = ", pvalue)
        
            
index100Path = "Data/Univ/Index_NIFTY 100.xlsx"
index50Path = "Data/Univ/Index_NIFTY 50.xlsx"
index50nPath = "Data/Univ/Index_NIFTY NEXT 50.xlsx"
stockPath = "Data"

#To save dataframes for first run
""" 
sd = save_data(index100Path, index50Path, index50nPath, stockPath)
tickers = sd.make_dataframe()
"""

ts = stocksData()
st = stats_tests()
stock_dict, tickers = ts.fetch_data("Saved")

factor = Factor.Momentum(stock_dict, tickers)
momentum_returns = factor.momentum_return(priceType="Open", method="last", 
                                freq="W", period=12, skip=1)

#one hot encoded portfolio 1=part of top decile
top_returns = factor.top_n_percentile(top_n=10)
#portfolio returns
portfolio_returns = factor.compute_prtfolio_returns(top_returns, 10, 12)
plots.portfolio_returns_plot(portfolio_returns)

#Performing KS test on portfolio returns
st.ks_test(portfolio_returns)
