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

class iim_factors:
    """
    get factors from IIMA dataset
    """
    
    def __init__(self, factor_location):
        self.factors = pd.DataFrame()
        self.file = factor_location
        
    def get_factors(self):
        df = pd.read_csv(self.file)
        df["Month"] = pd.to_datetime(df["Month"], format='%Y%m')
        df.set_index('Month', inplace=True)
        return df
        
    
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
        
        #initialize member variables
        def __init__(self, stocks, tickers):
            self.pricesDict = {}
            self.momentum = {}
            self.prices = pd.DataFrame()
            self.momentums = pd.DataFrame()
            self.stocks = stocks
            self.tickers = tickers
            
        #return log returns based on open/close/etc, method of mean/end, number of periods and a skip period
        def momentum_return(self, priceType, method="last", freq="1M", period=12, skip=1):
            for ticker in self.stocks:
                df = self.stocks[ticker]
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
                if (method == "last"):
                    prices_df = df.resample(freq).last()
                if (method == "mean"):
                    prices_df = df.resample(freq).mean()
                if (method == "first"):
                    prices_df = df.resample(freq).first()
                self.pricesDict[ticker] = prices_df[priceType]
            self.prices = pd.concat((self.pricesDict).values(), axis=1)
            self.prices.columns = self.tickers
            self.momentums = np.log(self.prices.shift(skip)/self.prices.shift(period))
            return self.momentums
        
        #return a one hot encoded matrix with 1 for top n percentile tickers for each period
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
        
        #for generating weights --> future purposes, rn equal wts at unity
        def generate_weights(self):
            weights_df = pd.DataFrame(1, index=self.momentums.index, columns=self.tickers)
            return weights_df
        
        #compute the returns given long and short
        def compute_prtfolio_returns(self, one_hot, n_stocks, period, weights):
            self.momentum["P_ret"] = (self.momentums*one_hot*weights).sum(axis=1)/n_stocks
            return self.momentum["P_ret"][period:]
    
class stats_tests:
    
    #Perform KS test wrt normal distribution
    def ks_test(self, x):
        mean = x.mean()
        std = x.std(ddof=0)
        kstat, pvalue = kstest(x.values, "norm", args=(mean,std),
                               alternative="two-sided", mode='approx')
        print("ks statistic = ", kstat)
        print("p value = ", pvalue)
        
class Model:
    
    def __init__(self, factors, p_returns):
        self.factors = factors
        self.p_returns = p_returns
        
    def factor_returns(self, rm=True, SMB=False, HML=False, WML=False, freq="1M", method="mean"):
        resampled_factors = self.factors.resample(freq).mean()
        if freq=="1M":
            factor_dates_years = [y.year for y in resampled_factors.index]
            momentum_dates_years = [y.year for y in self.p_returns.index]
            incl = []
            excl = []
            for i in factor_dates_years:
                incl.append(i in list(set(momentum_dates_years)))
            for i in momentum_dates_years:
                excl.append(i in list(set(factor_dates_years)))
            required_factors = self.factors[np.array(incl)]
            required_returns = self.p_returns[np.array(excl)]
        return required_factors["Rf %"].values, required_factors["Rm-Rf %"].values, required_returns.values
    
    def Regression(self, Rf, excess_market, portfolio_returns):
        Y = np.array(excess_market)
        X = np.array(portfolio_returns)-np.array(Rf)
        X = sm.add_constant(X)
        reg = sm.OLS(Y, X).fit()
        return reg
        
        

index100Path = "Data/Univ/Index_NIFTY 100.xlsx"
index50Path = "Data/Univ/Index_NIFTY 50.xlsx"
index50nPath = "Data/Univ/Index_NIFTY NEXT 50.xlsx"
stockPath = "Data"

ts = stocksData()
st = stats_tests()
stock_dict, tickers = ts.fetch_data("Saved")

factor = Factor.Momentum(stock_dict, tickers)
momentum_returns = factor.momentum_return(priceType="Open", method="last", 
                                freq="1M", period=12, skip=1)

top_returns = factor.top_n_percentile(top_n=10)
weight_df = factor.generate_weights()
portfolio_returns = factor.compute_prtfolio_returns(top_returns, 10, 12, weight_df)

FF = iim_factors("Data/Univ/FFreturns.csv")
FFdf = FF.get_factors()

m = Model(FFdf, portfolio_returns)
Rf, excess_market, portfolio_returns = m.factor_returns()
reg = m.Regression(Rf, excess_market, portfolio_returns)

for attributeIndex in range (0, 2):
    print(reg.pvalues[attributeIndex])
