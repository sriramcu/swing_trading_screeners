import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from warnings import simplefilter


# Constants for screening criteria
MIN_ABOVE_EMA20 = 0.98
EMA_DIFF_PERCENT = 0.005  # minimum difference between 10EMA and 20EMA
PRICE_EMA10_TOLERANCE = 0.04  # maximum tolerance above 10EMA (to avoid buying overpriced stocks)
MIN_PRICE_ABOVE_SMA200 = 0.01  # minimum distance above 200 SMA
MAX_RSI = 67  # Maximum allowed RSI value
MAX_STOCK_LIST_LIMIT = 10000

# Other constants
MIN_DATA_DAYS = 200  # Minimum days of data required
EMA10_PERIOD = 10
EMA20_PERIOD = 20
SMA50_PERIOD = 50
SMA200_PERIOD = 200
RSI_PERIOD = 14

# MACD constants
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SKIP_MACD = True

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def fetch_indian_stocks():
    """Read Nifty 200 stocks from CSV and append '.NS' suffix"""
    try:
        df = pd.read_csv('ind_nifty200list.csv')
        symbols = df['Symbol'].tolist()
        return [f"{symbol}.NS" for symbol in symbols]
    except FileNotFoundError:
        raise SystemExit("Error: CSV file 'ind_nifty200list.csv' not found")
    except KeyError:
        raise SystemExit("Error: CSV file must contain a 'Symbol' column")


def check_stock_criteria(data, stock):
    """Apply the new screening criteria to a stock"""
    close_prices = data[(stock, 'Close')].dropna()

    # Check if we have enough data
    if len(close_prices) < MIN_DATA_DAYS:
        return False

    # Calculate EMAs
    ema10 = close_prices.ewm(span=EMA10_PERIOD, adjust=False).mean()
    ema20 = close_prices.ewm(span=EMA20_PERIOD, adjust=False).mean()

    # Calculate SMAs
    sma50 = close_prices.rolling(SMA50_PERIOD).mean()
    sma200 = close_prices.rolling(SMA200_PERIOD).mean()

    # Calculate RSI
    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate MACD
    ema_fast = close_prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close_prices.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()

    # Get latest values
    latest_close = close_prices.iloc[-1]
    latest_ema10 = ema10.iloc[-1]
    latest_ema20 = ema20.iloc[-1]
    latest_sma50 = sma50.iloc[-1]
    latest_sma200 = sma200.iloc[-1]
    latest_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 100
    latest_macd = macd.iloc[-1]
    latest_macd_signal = macd_signal.iloc[-1]

    # Check all conditions using defined constants
    condition1 = (latest_ema10 - latest_ema20) / latest_ema20 >= EMA_DIFF_PERCENT
    condition2 = latest_close > latest_ema20 * MIN_ABOVE_EMA20
    condition3 = latest_close <= latest_ema10 * (1 + PRICE_EMA10_TOLERANCE)
    condition4 = (latest_close > latest_sma50 and
                  latest_ema10 > latest_sma50 and
                  latest_ema20 > latest_sma50 and
                  latest_sma50 > latest_sma200)
    condition5 = latest_close >= latest_sma200 * (1 + MIN_PRICE_ABOVE_SMA200)
    condition6 = latest_rsi < MAX_RSI
    condition7 = latest_macd > latest_macd_signal or SKIP_MACD

    return all([condition1, condition2, condition3, condition4, condition5, condition6, condition7])


def get_filtered_stocks(verbose=False):
    """Fetch and filter Indian stocks based on criteria"""
    stocks = fetch_indian_stocks()
    stocks = stocks[:MAX_STOCK_LIST_LIMIT]

    # Date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    if verbose:
        print(f"Fetching data for {len(stocks)} stocks...")

    # Download all stock data at once
    data = yf.download(stocks, start=start_date, end=end_date, group_by='ticker')

    if data.empty:
        raise ValueError("No data returned from Yahoo Finance API")

    if verbose:
        print("Data download complete. Applying screening criteria...")
        print(list(data))

    # Filter stocks
    selected_stocks = []
    for stock in stocks:
        try:
            # Skip if data is missing for this stock
            if (stock, 'Close') not in data.columns:
                if verbose:
                    print(f"Data missing for {stock}. Skipping...")
                continue

            if check_stock_criteria(data, stock):
                selected_stocks.append(stock)
        except Exception as e:
            if verbose:
                print(f"Error processing {stock}: {str(e)}")

    return selected_stocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen Indian stocks using technical criteria')
    parser.add_argument(    '-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    try:
        filtered_stocks = get_filtered_stocks(verbose=args.verbose)
        print("\nStocks meeting all criteria:")
        for stock in filtered_stocks:
            print(stock)
        print(f"\nTotal stocks meeting criteria: {len(filtered_stocks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
