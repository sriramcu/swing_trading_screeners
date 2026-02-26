# Swing Trading Screeners (US + India)

A small collection of **stock screening scripts** built around Yahoo Finance data (`yfinance`).
**Current / primary screener:** `13_21_logic.py` (EMA 13/21 + volume spike + recent-breakout filter + caching + symbol-range selection).

> ✅ This repository is **screening-only**.  
> It does **not** place trades, backtest, or paper trade.

---

## TL;DR (quick start)

If you have **zero coding experience**, do this:

1) **Download / clone** this repo  
2) Open a terminal in the repo folder  
3) Install dependencies:
```bash
pip install yfinance pandas numpy
```

4) Run the screener (defaults to **US + `us_stocks.csv` + top 200**):
```bash
python 13_21_logic.py
```

Common runs:

- **India (Nifty 200 list)**:
```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv
```

- **US, top 500 instead of top 200**:
```bash
python 13_21_logic.py --limit 500
```

- **Rank slice (e.g., 2500–2700 from a big ranked CSV)**:
```bash
python 13_21_logic.py --country us --csv us_stocks.csv --rank-start 2500 --rank-end 2700
```

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
--lower-close-ratio
```

Example:

```
--lower-close-ratio 0.97
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

### Screen US stocks (default run)

If you run with **no flags**, the script uses defaults:

- `--country us`
- `--csv us_stocks.csv`
- Universe selection defaults to **top 200** when `--limit` and `--rank-*` are not provided

```bash
python 13_21_logic.py
```

### Screen Indian stocks

```bash
python 13_21_logic.py --country india --csv ind_nifty200list.csv
```

Example output:

```
[CACHE] Using latest cache: 20260219_country-india_n-0200_h-bf39c1aa_s-00.pkl
[DEBUG USED CLOSE DATE] RELIANCE.NS: using last_date=2026-02-18 (exchange_tz=Asia/Kolkata, exchange_today=2026-02-19, utc_now=2026-02-19T15:44:23+00:00)

=== RESULTS ===
Country: india
Screened: 200 tickers
Passed: 4

TRENT.NS               EarningsDate=2026-02-05
DMART.NS               EarningsDate=2026-01-28
TITAN.NS               EarningsDate=2026-02-06
HDFCBANK.NS            EarningsDate=2026-01-31
```

---

### Screen US stocks (explicit)

```bash
python 13_21_logic.py --country us --csv us_stocks.csv
```

Example output:

```
[CACHE] Downloaded fresh data → wrote: 20260219_country-us_n-0300_h-ac8819e2_s-01.pkl
[DEBUG USED CLOSE DATE] AAPL: using last_date=2026-02-18 (exchange_tz=America/New_York, exchange_today=2026-02-19, utc_now=2026-02-19T15:44:23+00:00)

=== RESULTS ===
Country: us
Screened: 300 tickers
Passed: 3

NVDA                   EarningsDate=2026-02-20
META                   EarningsDate=2026-01-30
AMZN                   EarningsDate=2026-02-08
```

---

## Selecting a specific symbol range

Useful when working with ranked CSV lists.

Example: screen stocks ranked 2500–2700 (1-based, inclusive)

```
python 13_21_logic.py   --country us   --csv us_stocks.csv   --rank-start 2500   --rank-end 2700
```

Notes:
- `--rank-start` / `--rank-end` are **1-based and inclusive**.
- If you provide only `--rank-start`, it selects from that rank to the end.

---

## Command-line arguments (complete guide)

### Common arguments (most people only need these)

* `--country {us,india}` (default: `us`)  
  Market to screen.

* `--csv PATH` (default: `us_stocks.csv`)  
  CSV file containing symbols.

* `--symbol-col NAME` (default: `Symbol`)  
  Column name containing tickers.

Examples:

```bash
python 13_21_logic.py --country us --csv stocks.csv --symbol-col Ticker
python 13_21_logic.py --country india --csv ind_nifty500list.csv
```

---

### Universe selection

You can select the symbol universe in **two ways**:

#### A) Top-N (easy)
* `--limit N` (default: None; but if *everything* is None → defaults to top 200)

```bash
python 13_21_logic.py --limit 500
```

If you pass a huge number (e.g., `--limit 10000000`), Python slicing will just take “all available symbols in the CSV” (no error).

#### B) Rank range (recommended for big ranked CSVs)
* `--rank-start N` (1-based, inclusive)
* `--rank-end N` (1-based, inclusive)

```bash
python 13_21_logic.py --rank-start 2500 --rank-end 2700
```

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
* `--lower-close-ratio X` (default: 1.0)

Examples:

```bash
python 13_21_logic.py --country us --csv us_stocks.csv --vol-mult 1.2
```

More relaxed EMA:

```bash
python 13_21_logic.py   --country us   --csv us_stocks.csv   --lower-close-ratio 0.98
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
  last_date: 2026-02-18 00:00:00
  latest_close: 884.21
  latest_open: 875.44
  ema_fast: 891.02
  ema_slow: 903.11
  avg_vol: 45.2M
  latest_vol: 82.4M
  bars_since_not_aboveBoth_strict: 2
  within_recency_strict: True
```

---

## Timezone resiliency / excluding the ongoing trading session (important)

Sometimes **yfinance includes a forming daily bar for “today”** (especially for US tickers during market hours).  
If you use that bar as “latest close”, you get inconsistent results intraday.

**Fix implemented in `13_21_logic.py`:**
- If the market is **currently open** (country-aware, exchange timezone), and the latest row is dated “today” in that exchange timezone, the script **drops that last row** before screening.
- This means the screener always uses the last **completed** daily close.

To make this visible, the script prints a one-time line like:

```
[DEBUG USED CLOSE DATE] AAPL: using last_date=2026-02-18 (exchange_tz=America/New_York, exchange_today=2026-02-19, utc_now=2026-02-19T15:44:23+00:00)
```

If something ever changes in yfinance behavior, this debug line should make it obvious what date your screener is actually using.

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
* number of tickers selected
* a short hash of the **actual selected symbol subset** (prevents collisions when two different ranges have the same count)

Example filename:

```
20260219_country-us_n-0200_h-bf39c1aa_s-00.pkl
```

### ELI5 caching explanation

Think of the cache like this:

- The first run today says: “I’m screening **these exact 200 tickers** for **US** today.”
- It downloads all prices once, and saves them in `.cache_yf/prices/`.
- Later runs today with the **same exact ticker subset** reuse that file instead of hammering Yahoo Finance again.

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
[CACHE] Using explicit file: 20260219_country-us_n-0200_h-bf39c1aa_s-01.pkl
```

---

## Example debug session (illustrative)

```
[INFO] No --limit or --rank-range provided. Defaulting to top 200 symbols.
[CACHE] Using latest cache: 20260219_country-us_n-0200_h-bf39c1aa_s-00.pkl
[DEBUG USED CLOSE DATE] NVDA: using last_date=2026-02-18 (exchange_tz=America/New_York, exchange_today=2026-02-19, utc_now=2026-02-19T15:44:23+00:00)

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
* Cache keys include ticker count and a subset hash, so changing symbol ranges creates separate caches.
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
