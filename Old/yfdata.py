# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:21:59 2021

@author: Aditya
"""

import pandas as pd
import yfinance as yf

#get_data from wtage file, index as string, generalize variable name
#call it security universe
#tickers get a seperate function
#modularize
def get_data_NSE(start_date, end_date, priceType, index=100):
    stocks50 = pd.read_csv("Data/Univ/Nifty_Constituent_Weightage.csv")
    stocks100 = pd.read_csv("Data/Univ/Nifty_100_Constituent_Weightage.csv")
    #delete
    if(index==100):
        stocks = stocks100
    else:
        stocks = stocks50
    all_tickers = stocks.columns.values[1:]
    data = pd.DataFrame()
    for ticker in all_tickers:
        search = ticker + ".NS"
        print("fetching ",search)
        data[ticker] = yf.download(search, start_date, end_date)[priceType]
        print(data[ticker].head(5))
    return data 
