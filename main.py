# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:18:36 2021

@author: Aditya
"""

from saveData import save_data
from pandas.tseries.offsets import MonthEnd
import datetime as dt
import plots
import pandas as pd
import os
import h5py
import numpy as np
from scipy.stats import kstest
import statsmodels.api as sm
from collections import defaultdict

class iim_factors(object):
    """
    get factors from IIMA dataset
    """
    
    def __init__(self, factor_location):
        self.iimfactors = pd.DataFrame()
        self.file = factor_location
        
    def get_factors(self):
        df = pd.read_csv(self.file)
        df["Month"] = pd.to_datetime(df["Month"], format='%Y%m')
        df.set_index('Month', inplace=True)
        self.iimfactors = df        
    
class stocksData(object):
    """
    class for fetching individual ticker data from HDF5 files
    Generates a dictionary of [ticker]->daily returns
    Combines individual stock data in the dictionary to one single dataframe
    stocksDF = timestamp -> price (open/close etc as per argument priceType)
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

class prices(stocksData):
    """
    child class of stocksData that will contain prices of various kinds
    """
    
    def __init__(self, sd):
        self.stocks = sd.stocks.copy()
        self.tickers = sd.tickers.copy()
        self.pricesDf = pd.DataFrame()
        
    #create one dataframe for all stock and return it
    #take pricetype as argument to filter by price type
    #ALso use this to get open prices for next trading day
    def create_prices_df(self, priceType):
        flag = True
        for ticker in self.stocks:
            df = self.stocks[ticker].copy()
            df.set_index('Date', inplace=True)
            df = df[priceType]
            df = df[~df.index.duplicated()]
            #rename the series name to ticker name to preserve order later
            df.name = ticker
            #set first dataframe as first pd series
            if(flag):
                pricesDf = df
                flag = False
            #concat remaining series
            else:
                pricesDf = pd.concat([pricesDf, df], join="outer", axis=1)
            self.pricesDf = pricesDf
        return pricesDf

#a function to sort the tickers for each timestamp by descending value
#then return a dictionary of n bins of tickers in order
def sort_by_col(df, n):
    sorted_dict = defaultdict(list)
    for date,tickers in df.iterrows():
        ticker_by_desc_order = tickers.sort_values(ascending=False)
        for i in range(0,100,n):
            sorted_dict[tickers.name].append(ticker_by_desc_order[i:i+n])
    return sorted_dict

class momentum(prices):
    """
    class for getting momentum of various kinds
    inherits pricesDf from stocksData class for analysis
    """
    
    #initialize member variables from stocksData
    def __init__(self, p):
        self.pricesDf = p.pricesDf
        self.momentums = pd.DataFrame()

    #resample the data given frequency
    def resample_returns(self, method="last", freq="M"):
        if(method=="last"):
            self.pricesDf = self.pricesDf.groupby(pd.Grouper(freq=freq)).tail(1)
        if(method=="first"):
            self.pricesDf = self.pricesDf.groupby(pd.Grouper(freq=freq)).head(1)
        if(method=="mean"):
            self.pricesDf = self.pricesDf.resample(freq).mean()
        return self.pricesDf
            
    #return log returns based on open/close/etc, method of mean/end, number of periods and a skip period
    def momentum_return(self, period=12, skip=1):
        #compute momentum from returns
        self.momentums = np.log(self.pricesDf.shift(skip)/self.pricesDf.shift(period))
        return self.momentums
    
    #compute return using 2 dataframes. eg, overnight returns between close and open
    def overnight_returns(self, df2):
        shiftedPrices = self.pricesDf.shift(1).copy()
        shiftedPrices = shiftedPrices.set_index(df2.index)
        return np.log(df2/shiftedPrices)
      
    #get the n bins of prices and chose the first bin (topn) for each time stamp
    #create a data frame with top n returns for each timestamp
    def top_n_percentile_returns(self, top_n):
        top_n_df = pd.DataFrame()
        top_n_tickers = pd.DataFrame()
        sorted_dict = sort_by_col(self.momentums,top_n)
        for timestamp in sorted_dict:
            top_n_df[timestamp] = sorted_dict[timestamp][0].values
            top_n_tickers[timestamp] = sorted_dict[timestamp][0].index
        #top_n_df has the returns of n winning stocks desc order
        #also return winning return preserving order
        top_n_df = (top_n_df.T).dropna()
        top_n_tickers = top_n_tickers[top_n_df.index].T
        return top_n_df, top_n_tickers

class execute(object):
    """
    Given a list of tickers to long or short, exexute portfolio created
    Get returns using open and close prices after holding period
    """
    def __init__(self, tickerList, opendf, closedf):
        self.tickerList = tickerList
        self.opendf = opendf
        self.closedf = closedf
        
    #using the momentum returns at t-period, execute portfolio at t
    #held for holding period
    #reindex to match timestamps for division
    #then switch to month end index
    def rebalance_portfolio(self, period=1, holding=1):
        reindex_close = self.closedf.copy()
        reindex_close = reindex_close.set_index(self.opendf.index).shift(holding-1)
        realized_return = np.log(reindex_close/self.opendf).set_index(self.closedf.index)
        #because portfolio is implemented in the next period shift
        top_tickers_ = top_tickers.shift(period).dropna()
        timestamp_returns = defaultdict()
        for i,row in top_tickers_.iterrows():
            timestamp = row.name
            tickers = row.values
            timestamp_returns[timestamp] = realized_return.loc[timestamp][tickers].sum()
        timestamp_returns = pd.Series(timestamp_returns)
        plots.portfolio_returns_plot(timestamp_returns)
    
class stats_tests:
    
    #Perform KS test wrt normal distribution
    def ks_test(self, x):
        mean = x.mean()
        std = x.std(ddof=0)
        kstat, pvalue = kstest(x.values, "norm", args=(mean,std),
                               alternative="two-sided", mode='approx')
        print("ks statistic = ", kstat)
        print("p value = ", pvalue)
        
class Model(object):
    """
    Parent class for all models
    Variables are factors (X) and portfolio returns (Y)
    """
    def __init__(self, factors, returns):
        self.factors = factors
        self.returns = returns

    #since factors and returns can be measured at different frequencies etc
    #resample_factors will match the timestamps and return a singl dataframe
    #of X and Y for each timestamp

class iim_factor_analysis(iim_factors):
    """
    class to implement the method provided in the paper
    inherits iim factor dataframe
    """
    
    def __init__(self, iim, returns):
        self.factors = iim.iim_factors
        self.returns = returns
    
    #since factors and returns are at different frequencies
    #matching them with iim factors timestamps

index100Path = "Data/Univ/Index_NIFTY 100.xlsx"
index50Path = "Data/Univ/Index_NIFTY 50.xlsx"
index50nPath = "Data/Univ/Index_NIFTY NEXT 50.xlsx"
stockPath = "Data"

sd = stocksData()
s, t = sd.fetch_data("Saved")

cp = prices(sd)
op = prices(sd)

closePricesDf = cp.create_prices_df("Close")
openPricesDf = op.create_prices_df("Open")

iim = iim_factors("Data/Univ/FFreturns.csv")
iim.get_factors()

opMomentum = momentum(op)
cpMomentum = momentum(cp)

openPricesDf = opMomentum.resample_returns("first","M")
closePricesDf = cpMomentum.resample_returns("last","M")

closePriceMomentum = cpMomentum.momentum_return(12,1)
overnight_returns = cpMomentum.overnight_returns(openPricesDf)

top_performers, top_tickers = cpMomentum.top_n_percentile_returns(10)

iim_portfolio = execute(top_tickers, openPricesDf, closePricesDf)
iim_portfolio.rebalance_portfolio()