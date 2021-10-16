# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:23:14 2021

@author: Aditya
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:18:36 2021

@author: Aditya
"""
#https://ssrn.com/abstract=3510433

from saveData import save_data
import yfdata
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

#move to seperate file
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

class yfStocksData:
    
    def __init__(self, start_date, end_date, priceType):
        self.prices = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.priceType = priceType
    
    #download yahoo finance data
    def get_data(self):
        df = yfdata.get_data_NSE(self.start_date, self.end_date, self.priceType, 100) 
        df.to_csv("yfdata"+self.priceType+".csv")
        self.prices = df
        return df
    
    #if already downloaded, get data from csv
    #change variable name
    #dont need pricetype
    def get_downloaded_data(self):
        self.prices = pd.read_csv("yfdata"+self.priceType+".csv")
        self.prices["Date"] = pd.to_datetime(self.prices["Date"], )
        return self.prices
    
    #get tickers that have 0 wt
    #add comments
    def index_tickers(self, wts):
        w = wts.copy()
        w["threshold"] = 0
        wts1 = w.drop('threshold', 1).eq(w['threshold'], 0)
        wts1 = wts1.apply(lambda x: ', '.join(x.index[x]),axis=1)
        tickers_dict = defaultdict(list)
        for date,ticker in zip(wts1.index,wts1):
            tickers_dict[date] = ticker.split(", ")
        return tickers_dict 
    
class yfPrices(yfStocksData):
    
    def  __init__(self, yf):
        self.prices = yf.prices
    
    #remove tickers whose wt is 0 and preserve all others
    #filter by dates from wt csv file then use it to find relev prices
    def relevant_prices(self, wts, start, end, freq="M"):
        #resample according to weights frequency
        reindex_price = self.prices.set_index(["Date"])
        reindex_price = reindex_price.groupby(pd.Grouper(freq=freq)).tail(1)
        #get only required timeframe
        reindex_price = reindex_price[start:end]
        rprices = reindex_price.copy()
        #due to discripancy in last trading days, reset index to integer
        rprices.reset_index(inplace=True)
        print("reindex", rprices.tail(10), sep="\n")
        for timestamp in wts.keys():
            tickers = wts[timestamp]
            rprices.loc[timestamp+1,tickers] = np.nan
        return(rprices)
    
class factor:
    #a function to sort the tickers for each timestamp by descending value
    #then return a dictionary of n bins of tickers in order
    #have a prototype
    #have an init, having self.factor.name = price momentum, factortype = momentum
    def sort_by_col(self, df, n):
        sorted_dict = defaultdict(list)
        for date,tickers in df.iterrows():
            ticker_by_desc_order = tickers.sort_values(ascending=False)
            for i in range(0,100,n):
                sorted_dict[tickers.name].append(ticker_by_desc_order[i:i+n])
        return sorted_dict

class momentum(factor):
    """
    class for getting momentum of various kinds
    takes prices as an argument
    """
    #call things what they are
    #initialize member variables from given data
    #define build price momentum
    #variable name without plural
    #price_df feg
    def __init__(self, prices):
        self.pricesDf = prices
        #Price returns
        #dont initiate it here
        #compute price returns function within the class
        #self.price returns init to none then call the function
        #build momentum factor funtion (eg sorting)
        self.momentums = pd.DataFrame()

    #resample the data given frequency
    def resample_returns(self, method="last", freq="M"):
        self.pricesDf = self.pricesDf.set_index(["Date"])
        if(method=="last"):
            self.pricesDf = self.pricesDf.groupby(pd.Grouper(freq=freq)).tail(1)
        if(method=="first"):
            self.pricesDf = self.pricesDf.groupby(pd.Grouper(freq=freq)).head(1)
        if(method=="mean"):
            self.pricesDf = self.pricesDf.resample(freq).mean()
        self.pricesDf = self.pricesDf.dropna(axis=0, how="all")
        return self.pricesDf
            
    #return log returns based on open/close/etc, method of mean/end, number of periods and a skip period
    #log return function
    def momentum_return(self, period=12, skip=1):
        #compute momentum from returns
        self.momentums = np.log(self.pricesDf.shift(skip)/self.pricesDf.shift(period))
        #remove those cells with nan because they are not in index
        #tickers present in index
        wt = self.pricesDf.copy()
        wt = wt.replace(np.nan,0)
        wt[wt!=0] = 1
        #multiply with momentum to eliminate unnecessary returns
        self.momentums = self.momentums * wt
        self.momentums = self.momentums.replace(0, np.nan)
        return self.momentums
    
    #compute return using 2 dataframes. eg, overnight returns between close and open
    def overnight_returns(self, df2):
        shiftedPrices = self.pricesDf.shift(1).copy()
        shiftedPrices = shiftedPrices.set_index(df2.index)
        return np.log(df2/shiftedPrices)
      
    #get the n bins of prices and chose the first bin (topn) for each time stamp
    #create a data frame with top n returns for each timestamp
    #build price momentum
    def top_n_percentile_returns(self, top_n):
        top_n_df = pd.DataFrame()
        top_n_tickers = pd.DataFrame()
        sorted_dict = super().sort_by_col(self.momentums,top_n)
        for timestamp in sorted_dict:
            top_n_df[timestamp] = sorted_dict[timestamp][0].values
            top_n_tickers[timestamp] = sorted_dict[timestamp][0].index
        #top_n_df has the returns of n winning stocks desc order
        #also return winning return preserving order
        top_n_df = (top_n_df.T).dropna()
        top_n_tickers = top_n_tickers[top_n_df.index].T
        return top_n_df, top_n_tickers

#factor class
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
    #calc log returns
    def rebalance_portfolio(self, period=1, holding=1):
        print("month end closing prices:", self.closedf, sep="\n")
        #remove any excess data
        len_close = len(self.closedf)
        slice_open = self.opendf[:len_close]
        print("month start opening prices:", slice_open, sep="\n")
        reindex_close = self.closedf.copy()
        #since holding period on paper is one month, no need to shift
        #make abstract
        reindex_close = reindex_close.set_index(slice_open.index).shift(1-holding)
        realized_return = np.log(reindex_close/slice_open).set_index(self.closedf.index)*12
        #plotting average return
        print("realized monthly return:", realized_return, sep="\n")
        plots.portfolio_returns_plot(realized_return.mean(axis=1))
        #because portfolio is implemented in the next period shift
        #return Pi - Pi-1 /Pi
        top_tickers_ = top_tickers.shift(period).dropna()
        timestamp_returns = defaultdict()
        timestamp_mean_returns = defaultdict()
        for i,row in top_tickers_.iterrows():
            timestamp = row.name
            tickers = row.values
            timestamp_returns[timestamp] = realized_return.loc[timestamp][tickers]
            timestamp_mean_returns[timestamp] = realized_return.loc[timestamp][tickers].mean()
        timestamp_returns = pd.DataFrame.from_dict(timestamp_returns)
        timestamp_mean_returns = pd.Series(timestamp_mean_returns) * 12
        print("top portfolio returns", timestamp_mean_returns, "\n")
        plots.portfolio_returns_plot(timestamp_mean_returns)
        return realized_return, timestamp_returns
    
    def calculate_prices(self, portfolio_ret):
        pret = portfolio_ret.replace(np.nan, 0).T
        print(pret)
        monthly_rets = pret.sum(axis=1)
        print(monthly_rets)
    
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

 
start_date = "2013-05-01"
end_date = "2021-04-30"

yfopen = yfStocksData(start_date, end_date, "Open")
yfclose = yfStocksData(start_date, end_date, "Adj Close")                

#yfopen.get_data()
#yfclose.get_data()

dfopen = yfopen.get_downloaded_data()
dfclose = yfclose.get_downloaded_data()
print("Downloaded data", dfopen.head(10), sep="\n")

dfclosecopy = dfclose.copy()

wts = pd.read_csv("Data/Univ/Nifty_100_Constituent_Weightage.csv")
print("wts",wts.head(10),sep="\n")

tickers_dict = yfopen.index_tickers(wts)

yfopen = yfPrices(yfopen)
yfclose = yfPrices(yfclose)
dfclose = yfclose.relevant_prices(tickers_dict, wts.index.values[0], wts.index.values[-1])

openm = momentum(dfopen)
closem = momentum(dfclose)
closestartm = momentum(dfclosecopy)

resampleopen = openm.resample_returns(method="first")
resampleclose = closem.resample_returns()
resampleclosestart = closestartm.resample_returns(method="first")

closemomentum = closem.momentum_return()

top_n_df, top_tickers = closem.top_n_percentile_returns(10)

p = execute(top_tickers, resampleclosestart, resampleclose)
rr, tr = p.rebalance_portfolio()


