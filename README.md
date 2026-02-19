# Swing Trading Screeners (US + India)

A small collection of **stock screening scripts** built around Yahoo Finance data (`yfinance`).
**Current / primary screener:** `13_21_logic.py` (EMA 13/21 + volume spike + recent-breakout filter + caching + symbol-range selection).

> ✅ This repository is **screening-only**.
> It does **not** place trades, backtest, or paper trade.

---

## Directory structure

```
sriramcu-swing_trading_screeners/
├── 13_21_logic.py              # ✅ main/current screener (US + India) + caching + range selection
├── ind_nifty200list.csv        # India symbols list (expects column: Symbol)
├── ind_nifty500list.csv        # Larger India symbols list (expects column: Symbol)
├── us_stocks.csv               # US symbols list (expects column: Symbol)
├── screen_india_stocks.py      # older India screener (more indicators)
└── screen_us_stocks_hma.py     # older US screener (HMA / other criteria)
```

---

## CSV files (symbol universes)

### India CSVs

* `ind_nifty200list.csv`
* `ind_nifty500list.csv`

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

---

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

---

### 1. Trend alignment (with optional relaxed tolerance)

By default:

```
Close > EMA(13)
Close > EMA(21)
```

However, the screener supports a relaxed tolerance using:

```
--lower_close_ratio
```

Example:

```
--lower_close_ratio 0.97
```

This means:

```
Close must be above 97% of EMA values
```

Important nuance:

* This relaxation applies only to EMA comparison.
* The breakout timing ("recency") still uses strict EMA crossing logic.

This ensures:

* Early-stage breakouts are not missed
* Recency logic remains strict and accurate

---

### 2. Volume expansion

```
Latest volume > SMA(volume, lookback) × multiplier
```

Default:

```
volume > SMA(volume, 30) × 1.5
```  

The default value of multiplier is 1.5 times monthly average volume, can be set with the `--vol-mult` flag; and the 30 day default lookback can likewise be set with `--vol-lookback` as illustrated in the strategy parameters subsection of the command line arguments section.

---

### 3. Bullish candle

```
Close > Open
```

---

### 4. Recent breakout timing

Price must have moved above both EMAs recently:

```
bars_since(last close BELOW EMA13 or EMA21) must be between 1 and recency
```

Default:

```
recency = 3 bars
```

This avoids:

* chasing extended trends
* late-stage entries

---

### All conditions must be satisfied on the most recent candle.

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

Example output:

```
[CACHE] Using latest cache: 20260219_country-india_n-0200_s-00.pkl
[CACHE HASH] key=20260219|india|200 → hash=bf39c1aa

=== RESULTS ===
Country: india
Screened: 200 tickers
Passed: 4

TRENT.NS               05-02-2026
DMART.NS               28-01-2026
TITAN.NS               06-02-2026
HDFCBANK.NS           31-01-2026
```

---

### Screen US stocks

```bash
python 13_21_logic.py --country us --csv us_stocks.csv
```

Example output:

```
[CACHE] Downloaded fresh data → wrote: 20260219_country-us_n-0300_s-01.pkl
[CACHE HASH] key=20260219|us|300 → hash=ac8819e2

=== RESULTS ===
Country: us
Screened: 300 tickers
Passed: 3

NVDA                   20-02-2026
META                   30-01-2026
AMZN                   08-02-2026
```

---

## Selecting a specific symbol range

Useful when working with ranked CSV lists.

Example: screen stocks ranked 2500–2700

```
python 13_21_logic.py \
  --country us \
  --csv us_stocks.csv \
  --start-index 2500 \
  --end-index 2700
```

Example debug output:

```
[RANGE] selecting symbols index 2500 → 2700
[RANGE] total selected = 200 symbols
[CACHE HASH] key=20260219|us|200 → hash=d13f01bc
```

This integrates correctly with caching.

Cache key automatically adjusts to the selected range size.

---

## Command-line arguments (complete guide)

### Required arguments

* `--country {us,india}`
  Market to screen.

* `--csv PATH`
  CSV file containing symbols.

---

### Symbol handling

* `--symbol-col NAME` (default: `Symbol`)

Example:

```bash
python 13_21_logic.py --country us --csv stocks.csv --symbol-col Ticker
```

---

### Symbol range selection

New feature:

* `--start-index N`
* `--end-index N`

Example:

```bash
python 13_21_logic.py --country india \
  --csv ind_nifty500list.csv \
  --start-index 200 \
  --end-index 400
```

Debug output example:

```
[RANGE] selected indices 200–400
[RANGE HASH] count=200
```

---

### Universe size (legacy)

* `--limit N` (default: 300)

If range is specified, limit is ignored.

---

### History window

* `--years N` (default: 2)

Example:

```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv --years 1
```

---

### Strategy parameters

Core parameters:

* `--ema-fast N` (default: 13)
* `--ema-slow N` (default: 21)
* `--vol-lookback N` (default: 30)
* `--vol-mult X` (default: 1.5)
* `--recency N` (default: 3)
* `--lower_close_ratio X` (default: 1.0)

Examples:

```bash
python 13_21_logic.py --country us --csv us_stocks.csv --vol-mult 1.2
```

More relaxed EMA:

```bash
python 13_21_logic.py \
  --country us \
  --csv us_stocks.csv \
  --lower_close_ratio 0.98
```

Example debug:

```
[DEBUG] NVDA
 close=884.21
 ema13=891.02
 ema21=903.11
 relaxed threshold (0.98)=873.20
 PASS: relaxed EMA satisfied
```

---

### Verbose output

```
--verbose
```

Example:

```
NVDA PASSES
  latest_close: 884.21
  latest_open: 875.44
  ema_fast: 891.02
  ema_slow: 903.11
  avg_vol: 45.2M
  latest_vol: 82.4M
  bars_since_not_aboveBoth: 2
  within_recency: True
```

---

## Caching system (important)

Downloaded price data is cached to avoid repeated Yahoo Finance calls.

Cache location:

```
.cache_yf/prices/
```

Cache files are keyed by:

* UTC date
* country
* number of tickers selected (including range selections)

Example filename:

```
20260219_country-us_n-0200_s-00.pkl
```

Example debug output:

```
[CACHE HASH] raw key:
 date=20260219
 country=us
 ticker_count=200

[CACHE HASH] computed hash:
 bf39c1aa04b7c92e

[CACHE] Using latest cache: 20260219_country-us_n-0200_s-00.pkl
```

---

### Cache modes

Default:

```
--cache-mode auto
```

Uses today's cache if available.

Force refresh:

```
--cache-mode new
```

Example:

```
[CACHE] Downloaded fresh data
[CACHE HASH] new hash=ac8819e2
```

---

### Using specific cache

```
--cache-serial N
```

or

```
--cache-file PATH
```

Example:

```
[CACHE] Using explicit file: 20260219_country-us_n-0200_s-01.pkl
[CACHE HASH MATCH VERIFIED]
```

---

## Example debug session (illustrative)

```
[RANGE] selected indices 2500–2700
[CACHE HASH] key=20260219|us|200 → hash=d13f01bc

Downloading prices...

Checking NVDA:
 close=884.21
 ema13=891.02
 ema21=903.11
 relaxed_threshold=873.20
 volume=82M
 avg_volume=41M
 bars_since_breakout=2
 PASS

Checking META:
 close=495.21
 ema13=502.88
 ema21=511.01
 relaxed_threshold=490.77
 FAIL (below relaxed EMA)
```

---

## Notes / gotchas

* Yahoo Finance data can be missing for some symbols; those tickers are skipped.
* India tickers are converted to `.NS` unless the symbol already contains a suffix.
* Earnings dates come from `yf.Ticker(...).calendar` and may fail.
* Cache keys include ticker count, so changing symbol ranges creates separate caches.
* Recency applies to strict EMA crossing logic, not relaxed EMA logic.

---

## Older scripts (not the main focus)

* `screen_india_stocks.py`
* `screen_us_stocks_hma.py`

These are earlier experiments with more indicators and alternate criteria.
They remain for reference, but the **recommended workflow is `13_21_logic.py`.**

---

## Disclaimer

This project is for educational screening and experimentation only.
It is **not financial advice**.

