# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:18:36 2021

@author: Aditya
"""

import matplotlib.pyplot as plt
import seaborn as sns

def sample_ticker_price_plot(ticker, stockUniv):
    sns.set_style('whitegrid')
    close_prices = stockUniv[ticker]
    close_prices.plot(legend=True,figsize=(12,5))
    
def sample_ticker_returns_plot(ticker, stockUniv):
    sns.set_style('whitegrid')
    stockUniv[ticker].hist(bins=30)
    plt.xlabel("Monthly returns")
    plt.ylabel("Frequency")
    print(stockUniv[ticker].describe())
    
def portfolio_returns_plot(returns):
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(2,gridspec_kw={'height_ratios':[1.5,1]},figsize=(15,12))
    axs[0].plot(returns.index, returns.values)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Returns")
    sns.histplot(returns.values, ax=axs[1], bins=30, kde=True, common_norm=True)
    axs[1].set_xlabel("Returns")
    axs[1].set_ylabel("Frequency")
    print(returns.describe())
    
def topn_ts_plot(df):
    sns.set_style("darkgrid")
    sns.lineplot(data = df)