# Swing Trading Screeners (US + India)

A small collection of **stock screening scripts** built around Yahoo Finance data (`yfinance`).  
**Current / primary screener:** `13_21_logic.py` (EMA 13/21 + volume spike + recent-breakout filter).

> ✅ This repository is **screening-only**.  
> It does **not** place trades, backtest, or paper trade.

---

## Directory structure

```
sriramcu-swing_trading_screeners/
├── 13_21_logic.py              # ✅ main/current screener (US + India) + caching
├── ind_nifty200list.csv        # India symbols list (expects column: Symbol)
├── ind_nifty500list.csv        # Larger India symbols list (expects column: Symbol)
├── us_stocks.csv               # US symbols list (expects column: Symbol)
├── screen_india_stocks.py      # older India screener (more indicators)
└── screen_us_stocks_hma.py     # older US screener (HMA / other criteria)
```

---

## CSV files (symbol universes)

### India CSVs
- `ind_nifty200list.csv`
- `ind_nifty500list.csv`

Only one column is required:

```csv
Symbol
INFY
TCS
RELIANCE
```

Other columns may exist and are ignored.

When screening India (`--country india`), symbols are automatically converted:
```
INFY  →  INFY.NS
```

If a symbol already contains a suffix (`.NS`, `.BO`, etc.), it is left unchanged.

### US CSVs
Use `us_stocks.csv` with a similar format:

```csv
Symbol
AAPL
MSFT
NVDA
```

---

## The 13–21 screener (current strategy)

The file **`13_21_logic.py`** is the actively used screener.

It evaluates **only the most recent candle** for each ticker and checks:

1. **Trend alignment**
   - Price is above EMA(13)
   - Price is above EMA(21)

2. **Volume expansion**
   - Latest volume > SMA(volume, lookback) × multiplier

3. **Bullish candle**
   - Close > Open

4. **Recent breakout timing**
   - Price moved above both EMAs **within the last 1–3 trading bars**
   - This avoids chasing extended moves

All four conditions must be true for a ticker to pass.

The script then prints passing tickers and (best-effort) their earnings dates.

---

## Installation

```bash
pip install yfinance pandas numpy
```

---

## Basic usage

### Screen Indian stocks
```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv
```

### Screen US stocks
```bash
python 13_21_logic.py --country us --csv us_stocks.csv
```

---

## Command-line arguments (complete guide)

### Required arguments
- `--country {us,india}`  
  Market to screen.

- `--csv PATH`  
  CSV file containing symbols.

---

### Symbol handling
- `--symbol-col NAME` (default: `Symbol`)  
  Column name containing tickers.

Example:
```bash
python 13_21_logic.py --country us --csv stocks.csv --symbol-col Ticker
```

---

### Universe size
- `--limit N` (default: 300)  
  Only screen the first N symbols.

Example:
```bash
python 13_21_logic.py --country india --csv ind_nifty500list.csv --limit 100
```

---

### History window
- `--years N` (default: 2)  
  Number of years of data used for calculations.

Example:
```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv --years 1
```

---

### Strategy parameters
- `--ema-fast N` (default: 13)
- `--ema-slow N` (default: 21)
- `--vol-lookback N` (default: 30)
- `--vol-mult X` (default: 1.5)

Examples:
```bash
python 13_21_logic.py --country us --csv us_stocks.csv --vol-mult 2.0
python 13_21_logic.py --country us --csv us_stocks.csv --ema-fast 9 --ema-slow 21
```

---

### Verbose output
- `--verbose`  
  Prints indicator values for each passing ticker.

```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv --verbose
```

---

## Caching system (important)

Downloaded price data is cached to avoid repeated Yahoo Finance calls.

Cache location:
```
.cache_yf/prices/
```

Cache files are keyed by:
- UTC date
- country
- number of tickers

### Cache modes
- `--cache-mode auto` (default)  
  Use today’s cache if present; otherwise download and cache.

- `--cache-mode new`  
  Force a fresh download and write a new cache file.

Example:
```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv --cache-mode new
```

### Using a specific cache
- `--cache-serial N`  
  Use a specific cache serial from today.

```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv --cache-serial 1
```

- `--cache-file PATH`  
  Use an explicit cache file (country and ticker count must match).

```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv \
  --cache-file .cache_yf/prices/20260207_country-india_n-0200_s-00.pkl
```

---

## Notes / gotchas

- Yahoo Finance data can be missing for some symbols; those tickers are skipped.
- India tickers are converted to `.NS` unless the symbol already contains a suffix.
- Earnings dates come from `yf.Ticker(...).calendar` and may fail; the script prints a fallback message.

---

## Older scripts (not the main focus)

- `screen_india_stocks.py`
- `screen_us_stocks_hma.py`

These are earlier experiments with more indicators and alternate criteria.  
They remain for reference, but the **recommended workflow is `13_21_logic.py`.**

---

## Disclaimer

This project is for educational screening and experimentation only.  
It is **not financial advice**.
