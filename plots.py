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
    stockUniv[ticker].hist(bins=50)
    plt.xlabel("Monthly returns")
    plt.ylabel("Frequency")
    print(stockUniv[ticker].describe())
    