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

class stocks():
    
    def __init__(self):
        self.stocks = {}
        self.tickers = []
        self.monthends = {}
        self.momentum = {}
        self.prices = pd.DataFrame()
        self.momentums = pd.DataFrame()
        
    def fetch_data(self, filepath):
        for file in os.listdir(filepath):
            filedir = filepath+"/"+file
            f = h5py.File(filedir, 'r')
            key = list(f.keys())[0]
            self.stocks[key] = pd.read_hdf(path_or_buf=filedir, key=key, mode='r')
        self.tickers = list(self.stocks.keys())
        return self.stocks
    
    def find_monthends(self, type):
        for ticker in self.stocks:
            df = self.stocks[ticker]
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            monthend_df = df.resample('1M').last()
            self.monthends[ticker] = monthend_df[type]
        self.prices = pd.concat((self.monthends).values(), axis=1)
        self.prices.columns = self.tickers
        return self.prices
    
    def momentum_return(self):
        self.momentums = np.log(self.prices.shift(1)/self.prices.shift(12))
        return self.momentums
    
    def top_decile(self, top_n):
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
    
    def compute_prtfolio_returns(self, one_hot, n_stocks):
        self.momentum["P_ret"] = (self.momentums*one_hot).sum(axis=1)/n_stocks
        return self.momentum["P_ret"]
    
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

ts = stocks()
st = stats_tests()
stock_dict = ts.fetch_data("Saved")
prices = ts.find_monthends("Close")
momentums = ts.momentum_return()

#one hot encoded portfolio 1=part of top decile
top_returns = ts.top_decile(10)
#portfolio returns
portfolio_returns = ts.compute_prtfolio_returns(top_returns, 10)[12:]

plots.portfolio_returns_plot(portfolio_returns)

#Performing KS test on portfolio returns
st.ks_test(portfolio_returns)