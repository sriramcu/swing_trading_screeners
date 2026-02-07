import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from warnings import simplefilter
import argparse
from curl_cffi import requests

session = requests.Session(impersonate="chrome")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Constants for stock limits for both methods
STOCK_LIST_LIMIT = 100
MIDCAP_STOCK_LIST_LIMIT = 100

# Constants for HMA method
HMA_PERIOD = 30
TREND_DAYS = 3
MAX_PREMIUM_OVER_HMA = 1.035

# Constants for new criteria
MIN_ABOVE_EMA20 = 0.97
EMA10_MINUS_EMA20_MIN_FRACTION = 0.01 * (1/2)
MAX_PREMIUM_OVER_EMA10 = 1.05
MIN_PREMIUM_ABOVE_SMA200 = 1.1
MAX_RSI = 55

def fetch_top_stocks(verbose=False, stock_list_limit=STOCK_LIST_LIMIT):
    import csv
    stocks = []
    with open('us_stocks.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stocks.append(row['Symbol'])

    return_list = stocks[:stock_list_limit]
    if verbose:
        print("Top Stocks:", return_list)
    return return_list


def fetch_median_stocks(verbose=False, stock_list_limit=MIDCAP_STOCK_LIST_LIMIT):
    import csv
    stocks = []
    with open('us_stocks.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stocks.append(row['Symbol'])

    median_start = len(stocks) // 2 - stock_list_limit // 2
    median_end = median_start + stock_list_limit
    return_list = stocks[median_start:median_end]
    if verbose:
        print("Median Stocks:", return_list)
    return return_list


def calculate_hma(data, period):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    wma_half = data.rolling(window=half_length).mean()
    wma_full = data.rolling(window=period).mean()

    hma = 2 * wma_half - wma_full
    hma = hma.rolling(window=sqrt_length).mean()

    return hma


def check_stock_criteria(data, stock):
    data[('HMA', stock)] = calculate_hma(data[('Close', stock)], HMA_PERIOD)

    hma_trend = data[('HMA', stock)].diff().rolling(window=TREND_DAYS).apply(lambda x: all(x > 0), raw=True)

    latest_close = data[('Close', stock)].iloc[-1]
    latest_hma = data[('HMA', stock)].iloc[-1]

    price_within_hma = latest_close <= latest_hma * MAX_PREMIUM_OVER_HMA and latest_close >= latest_hma

    return hma_trend.iloc[-1] and price_within_hma


def check_stock_criteria_new(data, stock):
    if ('Close', stock) not in data.columns:
        return False
    close_prices = data[('Close', stock)].dropna()
    if len(close_prices) < 200:
        return False

    # Calculate EMAs
    ema10 = close_prices.ewm(span=10, adjust=False).mean()
    ema20 = close_prices.ewm(span=20, adjust=False).mean()

    # Calculate SMAs
    sma50 = close_prices.rolling(50).mean()
    sma200 = close_prices.rolling(200).mean()

    # Calculate RSI(14)
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    latest_close = close_prices.iloc[-1]
    latest_ema10 = ema10.iloc[-1]
    latest_ema20 = ema20.iloc[-1]
    latest_sma50 = sma50.iloc[-1]
    latest_sma200 = sma200.iloc[-1]
    latest_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 100  # Handle case where RSI is NaN

    condition1 = (latest_ema10 - latest_ema20) / latest_ema20 >= EMA10_MINUS_EMA20_MIN_FRACTION
    condition2 = latest_close > latest_ema20 * MIN_ABOVE_EMA20
    condition3 = latest_close <= latest_ema10 * MAX_PREMIUM_OVER_EMA10
    condition4 = (latest_close > latest_sma50 and latest_ema10 > latest_sma50 and
                  latest_ema20 > latest_sma50 and latest_sma50 > latest_sma200)
    condition5 = latest_close >= latest_sma200 * MIN_PREMIUM_ABOVE_SMA200
    condition6 = latest_rsi < MAX_RSI

    return condition1 and condition2 and condition3 and condition4 and condition5 and condition6


def get_filtered_stocks(method='new'):
    print(f"Using method: {method}")
    top_stocks = fetch_top_stocks()
    median_stocks = fetch_median_stocks()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    selected_top_stocks = []
    selected_median_stocks = []

    top_data = yf.download(top_stocks, start=start_date, end=end_date, session=session)
    median_data = yf.download(median_stocks, start=start_date, end=end_date, session=session)

    if top_data.empty or median_data.empty:
        raise ValueError("No data returned from Yahoo Finance API")

    check_func = check_stock_criteria if method == 'hma' else check_stock_criteria_new

    for stock in top_stocks:
        if ('Close', stock) not in top_data.columns:
            continue
        if check_func(top_data, stock):
            selected_top_stocks.append(stock)

    for stock in median_stocks:
        if ('Close', stock) not in median_data.columns:
            continue
        if check_func(median_data, stock):
            selected_median_stocks.append(stock)

    return selected_top_stocks, selected_median_stocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock screening tool.')
    parser.add_argument('--method', choices=['hma', 'new'], default='new',
                        help='Screening method to use: HMA (hma) or new criteria (new). Default is new.')
    args = parser.parse_args()

    filtered_top_stocks, filtered_median_stocks = get_filtered_stocks(method=args.method)
    print("Filtered Largecap Stocks:", filtered_top_stocks)
    print("Filtered Midcap Stocks:", filtered_median_stocks)
