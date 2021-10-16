# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:10:34 2021

@author: Aditya
"""

import pandas as pd
import yfinance as yf
import numpy as np
from collections import defaultdict
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns


####################   Helper functions  ######################################
def get_yahoo_finance_data(start_date, end_date, weights_data):
    """
    Functions to save yahoo finance data as csv
    Store all stock data in one dictionary in the form of "ticker_param"
    Save this as df (stock_universe) and stocks data.csv
    """
    tickers = weights_data.columns.values[1:]
    temp = pd.DataFrame()
    for ticker in tickers:
        stock_universe = pd.DataFrame()
        search = ticker + ".NS"
        print(ticker)
        ticker_df = yf.download(search, start_date, end_date)
        if not ticker_df.empty:
            stock_universe[ticker + "_Date"] = ticker_df.index.values
            stock_universe[ticker + "_Open"] = ticker_df["Open"].values
            stock_universe[ticker + "_High"] = ticker_df["High"].values
            stock_universe[ticker + "_Low"] = ticker_df["Close"].values
            stock_universe[ticker + "_Adj_close"] = ticker_df["Adj Close"].values
            stock_universe[ticker + "_Volume"] = ticker_df["Volume"].values
            temp = pd.concat([stock_universe, temp], axis=1)
            temp.to_csv("stocks data.csv")
    return temp

def get_index_data(file, price_type, freq):
    """
    function to fetch locally saved index data
    return prices and returns
    """
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    returns_df = df.resample(freq).ffill().pct_change()*100
    returns_df = returns_df[price_type].rename("nifty %")
    return df, returns_df


def get_local_data(path):
    """
    fetch local data
    """
    return pd.read_csv(path, low_memory=False)


def get_ticker_data(universe_data, ticker):
    """
    given a universe of stocks and a particular ticker
    get the ohlc table by stripping off ticker name from columns
    """
    Date = ticker + "_Date"
    Open = ticker + "_Open"
    High = ticker + "_High"
    Low = ticker + "_Low"
    AdjClose = ticker + "_Adj_close"
    Volume = ticker + "_Volume"
    stock_data = pd.DataFrame()
    if Date in universe_data.columns.values:
        stock_data["Date"] = pd.to_datetime(universe_data[Date])
        stock_data["Open"] = universe_data[Open]
        stock_data["High"] = universe_data[High]
        stock_data["Low"] = universe_data[Low]
        stock_data["AdjClose"] = universe_data[AdjClose]
        stock_data["Volume"] = universe_data[Volume]
        stock_data = stock_data.set_index("Date")
    else:
        """
        in case of empty data due to failed download, 
        return an empty ohlc table
        """
        return pd.DataFrame()
    return stock_data


def dict_stock_OHLC_tables(weights_data, universe_stocks):
    """
    Funtion to get dictionary of all available OHLC Tables
    Use weights data to get ticker name and search in local file
    """
    tickers = weights_data.columns.values[1:]
    dict_all_tables = {}
    for ticker in tickers:
        ticker_ohlc_table = get_ticker_data(universe_data, ticker)
        dict_all_tables[ticker] = ticker_ohlc_table
        if dict_all_tables[ticker].empty:
            del dict_all_tables[ticker]
    return dict_all_tables


def sort_dataframe_by_ticker(df, n):
    """
    Get sets of top performers in 0-n, n-2n, ... kn-100
    """
    sorted_dict = defaultdict(list)
    for date, tickers in df.iterrows():
        ticker_by_desc_order = tickers.sort_values(ascending=False)
        for i in range(0, 100, n):
            sorted_dict[tickers.name].append(ticker_by_desc_order[i:i + n])
    return sorted_dict

def rebalance_portfolio(top_tickers, returns, period=1, holding=1):
    # shift top tickers by period
    top_tickers = top_tickers.shift(1).dropna()
    # define a timestamp return dictionary
    timestamp_returns = {}
    # creating variable to store all returns
    selected_stock_returns = {}
    # iterate thru top tickers
    for i, row in top_tickers.iterrows():
        timestamp = row.name
        tickers = row.values
        selected_stock_returns[timestamp] = returns.loc[timestamp][tickers]
        timestamp_returns[timestamp] = returns.loc[timestamp][tickers].mean()
    tr = pd.Series(timestamp_returns).rename("tr")
    return tr, selected_stock_returns

def get_iim_factor_df(file):
    df = pd.read_csv(file)
    df["Month"] = pd.to_datetime(df["Month"], format='%Y%m')
    df = df.set_index("Month")
    df = df.resample("1M").first()
    return df

def merge_factors_df(**kwargs):
    dict_df = {}
    for key,value in kwargs.items():
        dict_df[key] = value
    dict_df["Rf %"] = 0
    return pd.DataFrame(dict_df).dropna()

def get_annual_df(df, percent):
    if(percent):
        df = 1 + df/100
    return df.resample("Y").prod() * 100

def pivot_returns(df, **kwargs):
    plot_df = {}
    for key, value in kwargs.items():
        plot_df[key] = df[value]
    merged_df = pd.DataFrame(plot_df).reset_index().dropna()
    merged_df["Year"] = pd.DatetimeIndex(merged_df["index"]).year
    merged_df["Month"] = pd.DatetimeIndex(merged_df["index"]).month
    pivot_df = merged_df.pivot("Year", "Month", ["portfolio", "market"])
    return pivot_df

def visualize_returns(df, **kwargs):
    plot_df = {}
    for key, value in kwargs.items():
        plot_df[key] = df[value]
    print(plot_df)
    n = len(plot_df)
    fig, ax = plt.subplots(n,1, figsize=(10,15))
    sns.set_theme()
    cmap = sns.diverging_palette(10,150, as_cmap=True)
    for num, key in enumerate(plot_df):
        res = sns.heatmap(data=plot_df[key], ax=ax[num], annot=True,
                          vmin=-20, vmax=20, center=0, cmap=cmap)
        res.axes.set_title(key,fontsize=26)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 24)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 22)
        res.set_xlabel("Month",fontsize=24)
        res.set_ylabel("Year",fontsize=24)
    fig.tight_layout(pad=2.0)
    plt.show()
    
def visualize_time_series_returns(df, marker='.', linestyle='none'):
    print(df)
    sns.set_theme()
    df.plot(marker=marker, linestyle=linestyle)
    plt.show()
####################   End of Helper functions  ###############################

class PriceReturns(object):
    """
    class for price and returns given a dictionary of all ohlc tables
    """

    def __init__(self, dict_all_tables):
        self.dict_all_tables = dict_all_tables

    def get_stock_ohlc_table(self, ticker):
        """
        get ohlc table from dictionary
        """
        return self.dict_all_tables[ticker]

    def get_stock_price(self, ohlc_table, price_type):
        """
        get price from ohlc table
        """
        return ohlc_table[price_type]

    def resample_prices(self, price, freq, method):
        """
        resample the price
        """
        if (method == "start"):
            return price.resample(freq, convention=method).first()
        if (method == "end"):
            return price.resample(freq).last()

    def generate_prices(self, price_type, freq, method):
        """
        return a dataframe of ticker->price
        """
        dict_universe_price_table = {}
        stock_price_df = pd.DataFrame()
        for ticker in self.dict_all_tables.keys():
            ohlc_table = self.get_stock_ohlc_table(ticker)
            price = self.get_stock_price(ohlc_table, price_type)
            resampled_price = self.resample_prices(price, freq, method)
            dict_universe_price_table[ticker] = resampled_price
            dict_universe_price_table[ticker] = dict_universe_price_table[
                ticker].rename(ticker)
        stock_price_df = pd.concat(dict_universe_price_table.values(), axis=1)
        return stock_price_df

    def calculate_stock_freq_return(self, ticker, price_type, freq="M"):
        """
        given ticker, pricetype, calculate frequency based return for that ticker
        Sum along axis=0 gives total return for frequency
        """
        ohlc_table = self.get_stock_ohlc_table(ticker)
        price = self.get_stock_price(ohlc_table, price_type).dropna()
        return price.resample(freq).ffill().pct_change()

    def generate_returns(self, price_type, freq):
        """
        generate returns for all tickers in universe and return it as a dictionary
        """
        dict_universe_returns_tables = {}
        for ticker in self.dict_all_tables.keys():
            dict_universe_returns_tables[ticker] = self.calculate_stock_freq_return(
                ticker, price_type, "M")
            dict_universe_returns_tables[ticker] = dict_universe_returns_tables[
                ticker].rename(ticker)
        return pd.concat(dict_universe_returns_tables.values(), axis=1)


###############################################################################
class FactorModel(object):
    """
    Class for implementing various factor models
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def linear_factor_model(self):
        reg = linear_model.LinearRegression().fit(self.X, self.Y)
        return reg
        

class CAPM(FactorModel):
    """
    Class for implementing CAPM
    """

    def __init__(self, factors, portfolio_return, market_return, rf):
        self.portfolio_excess_return = factors[portfolio_return] - factors[rf]
        self.portfolio_return = factors[portfolio_return]
        self.market_return = factors[market_return]
        self.rf = factors[rf]
        self.market_premium = self.market_return - self.rf    
        self.X = (self.market_return).values.reshape(-1,1)
        self.Y = self.portfolio_return.values.reshape(-1,1)
            
    def visualize_capm(self, reg):
        d = {"rm": self.market_return, "p": self.portfolio_return}
        df = pd.DataFrame(d)
        fig, ax = plt.subplots(figsize=(7,5))
        ax = sns.scatterplot(data=df, x="rm", y="p")
        ax.set_xlabel('Market return')
        ax.set_ylabel('Portfolio returns')
        ax.text(x=0.5, y=1.1, s='CAPM model', 
                fontsize=20, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
        ax.text(x=0.5, y=1.05, s='rP = alpha + beta(rm)', 
                fontsize=16, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
        plt.plot(self.X, reg.predict(self.X), color='red', linewidth=2)
        plt.show()

    def get_CAPM_coeff(self):
        reg = self.linear_factor_model()
        print("alpha = ", reg.intercept_)
        print("beta = ", reg.coef_)
        self.visualize_capm(reg)


###############################################################################


class Momentum(object):

    def __init__(self, dict_all_tables):
        self.price_momentum_dataframe = pd.DataFrame()
        self.dict_universe_price_momentums = {}
        self.dict_all_tables = dict_all_tables

    def calculate_stock_price_momentum(self, stock_prices, period=12, skip=1):
        """
        given returns and period, calculate price Momentum 
        """
        return np.log(stock_prices.shift(skip) / stock_prices.shift(period))

    def get_stock_ohlc_table(self, ticker):
        """
        given ticker, get ohlc table
        """
        return self.dict_all_tables[ticker]

    def resample_prices(self, price, freq):
        """
        resample based on frequency
        """
        return price.resample("M").last()

    def generate_universe_price_momentum(self, price_type, freq="M", period=12, skip=1):
        """
        get universe price momentum
        """
        for ticker in self.dict_all_tables.keys():
            ohlc_table = self.get_stock_ohlc_table(ticker)
            price = ohlc_table[price_type]
            resampled_price = self.resample_prices(price, freq)
            price_momentum = self.calculate_stock_price_momentum(
                resampled_price, period=12, skip=1)
            self.dict_universe_price_momentums[ticker] = price_momentum
            self.dict_universe_price_momentums[ticker] = self.dict_universe_price_momentums[
                ticker].rename(ticker)

        self.price_momentum_dataframe = pd.concat(
            self.dict_universe_price_momentums.values(), axis=1)
        self.price_momentum_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        return self.price_momentum_dataframe

    def top_n_percentile_price_momentum(self, n):
        sorted_dict = sort_dataframe_by_ticker(self.price_momentum_dataframe, n)
        top_n_price_momentum = pd.DataFrame()
        top_n_tickers = pd.DataFrame()
        for timestamp in sorted_dict:
            top_n_price_momentum[timestamp] = sorted_dict[timestamp][0].values
            top_n_tickers[timestamp] = sorted_dict[timestamp][0].index
        top_n_price_momentum = (top_n_price_momentum.T).dropna()
        top_n_tickers = top_n_tickers[top_n_price_momentum.index].T
        return top_n_price_momentum, top_n_tickers

    #Helper functions for visualization
    def get_price_momentum_and_price(self, price, ticker):
        return self.price_momentum_dataframe[ticker], price[ticker]

    def plot_price_momentum_with_price(self, ticker, price):
        """
        plot price momentum togethor with stock price of given ticker
        """
        ticker_price_momentum, ticker_price = self.get_price_momentum_and_price(price, ticker)
        sns.set_style('white')
        ax = ticker_price.dropna().plot(x="Month", y="Price", legend=False)
        ax2 = ax.twinx()
        ax2 = ticker_price_momentum.dropna().plot(x="Month", y="Return", legend=False, color="r")
        plt.show()

###############################################################################


# get data from yahoo finance
path = "stocks data.csv"
weights_data = pd.read_csv("Data/Univ/Nifty_100_Constituent_Weightage.csv")
index_df = get_index_data("Data/Univ/^CNX100.csv", "Adj Close", "M")

universe_data = get_local_data(path)
dict_all_tables = dict_stock_OHLC_tables(weights_data, universe_data)

pr = PriceReturns(dict_all_tables)

returns = pr.generate_returns("AdjClose", "M")

m = Momentum(dict_all_tables)
df = m.generate_universe_price_momentum("AdjClose")
top = m.top_n_percentile_price_momentum(10)

tr, sr = rebalance_portfolio(top[1], returns)

merged_df = merge_factors_df(portfolio=tr*100, market=index_df[1])
#merged_df = get_annual_df(merged_df, True)
capm = CAPM(merged_df, "portfolio", "market", "Rf %")
capm.get_CAPM_coeff()


pivot_df = pivot_returns(merged_df, portfolio="portfolio", market="market")
visualize_returns(pivot_df, portfolio="portfolio", market="market")
visualize_time_series_returns(merged_df, marker='.', linestyle='none')