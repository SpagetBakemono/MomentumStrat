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
stock_dict = ts.fetch_data("Saved")
prices = ts.find_monthends("Close")
momentums = ts.momentum_return()

print(prices.tail(5))
print(momentums.tail(5))

plots.sample_ticker_returns_plot("ACC", momentums)
