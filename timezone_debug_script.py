"""
Because sometimes yfinance shows ongoing trading session as the latest candlestick,
which should NOT be taken as the last "close" for filtering.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

sym = "DLTR"
df = yf.download(sym, period="10d", interval="1d", auto_adjust=False, progress=False)

print("UTC now:", datetime.now(timezone.utc))
print("Local now:", datetime.now())
print("Index dtype:", df.index.dtype)
print("Index tz:", getattr(df.index, "tz", None))
print(df.tail(5)[["Open","High","Low","Close","Volume"]])
print("Last index value repr:", repr(df.index[-1]))

last = df.tail(1)
print("Last row:")
print(last[["Open","High","Low","Close","Volume"]])

# A crude completeness check:
print("Is last bar volume zero?", float(last["Volume"].iloc[0]) == 0.0)
print("Is last bar OHLC identical?", float(last["Open"].iloc[0]) == float(last["High"].iloc[0]) == float(last["Low"].iloc[0]) == float(last["Close"].iloc[0]))

sym = "RELIANCE.NS"
df = yf.download(sym, period="10d", interval="1d", auto_adjust=False, progress=False)

from datetime import datetime, timezone
print("UTC now:", datetime.now(timezone.utc))
print("Local now:", datetime.now())
print("Index dtype:", df.index.dtype)
print("Index tz:", getattr(df.index, "tz", None))
print(df.tail(5)[["Open","High","Low","Close","Volume"]])
print("Last index value repr:", repr(df.index[-1]))

import yfinance as yf
import pandas as pd
print("yfinance version:", yf.__version__)
print("pandas version:", pd.__version__)