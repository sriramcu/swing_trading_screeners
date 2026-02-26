#!/usr/bin/env python3
"""
EMA + Volume Strike Zone Screener (US + India)

Screens tickers for the following "buy condition" on the latest candle:
- Green candle: Close > Open   (STRICT, uses raw close)
- Volume spike: Volume > SMA(Volume, lookback) * vol_multiplier  (STRICT, uses raw volume)
- Above both EMAs: Close > EMA(fast) AND Close > EMA(slow)
    * This can be RELAXED by a factor (e.g. 0.97 means allow Close > 0.97 * EMA)
    * IMPORTANT: Recency is still evaluated using STRICT close vs EMA (no relaxation)
- "Within N bars of breakout": barssince(not aboveBoth_strict) is 1..recency
- Data limited to last N years (download range + filter)

No trading, no backtesting, no paper trading. Screening only.

Key fixes vs your pasted version:
1) lower_close_ratio now relaxes EMA comparisons correctly:
   - before you did close = close * ratio, which makes EMA checks *stricter* (and also makes is_green stricter).
   - now we keep raw close for candle/volume/recency, and only relax EMA threshold:
        above_relaxed = close > ema * lower_close_ratio   (e.g., 0.97*ema)
2) Recency window is computed on STRICT above-both (raw close > raw EMA), as you wanted.
3) Added ability to select tickers by rank range from CSV (e.g., 2500-2700).
4) Cache key now includes a hash of the actual symbol subset so different ranges with same count won’t collide.
   (Still includes UTC date and num tickers as you requested.)

NEW (timezone resiliency / exclude ongoing session):
- yfinance can include a "forming" daily bar for the current trading day (especially US tickers intraday).
- This script now drops today's row ONLY when the market is currently open (country-aware),
  so screening always uses the last COMPLETED daily close.

Also prints a single debug line showing which date is being used for "latest close" (for one sample ticker).

Example:
  python screener.py --country india --csv ind_nifty200list.csv --rank-start 2500 --rank-end 2700
  python screener.py --country us --csv us_stocks.csv --limit 500 --cache-mode new --verbose
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import pickle
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import time as dtime

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# CSV + symbol handling
# ----------------------------
def read_symbols_from_csv(
    csv_path: str,
    symbol_col: str = "Symbol",
) -> list[str]:
    symbols: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or symbol_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain a '{symbol_col}' column. Found: {reader.fieldnames}")
        for row in reader:
            sym = (row.get(symbol_col) or "").strip()
            if sym:
                symbols.append(sym)
    return symbols


def slice_by_rank(
    symbols: list[str],
    *,
    limit: int | None,
    rank_start: int | None,
    rank_end: int | None,
) -> list[str]:
    """
    Apply either:
      - explicit rank range [rank_start, rank_end] (1-based, inclusive), OR
      - top N via limit

    NEW BEHAVIOR:
      - If rank_start, rank_end, AND limit are ALL None → default to top 200

    If rank_start is provided but rank_end is None -> take from rank_start to end.
    """

    DEFAULT_LIMIT = 200

    # If no rank range specified
    if rank_start is None and rank_end is None:

        # NEW: default to top 200
        if limit is None:
            limit = DEFAULT_LIMIT
            print(f"[INFO] No --limit or --rank-range provided. Defaulting to top {DEFAULT_LIMIT} symbols.")

        return symbols[: max(limit, 0)]

    # Rank range mode
    n = len(symbols)

    s = 1 if rank_start is None else rank_start
    e = n if rank_end is None else rank_end

    if s < 1:
        s = 1
    if e > n:
        e = n
    if e < s:
        return []

    return symbols[s - 1 : e]


def to_country_symbols(raw_symbols: list[str], country: str) -> list[str]:
    country = country.lower()
    if country in ("india", "in"):
        out = []
        for s in raw_symbols:
            s = s.strip()
            if not s:
                continue
            # If user already provided an exchange suffix, keep it
            if "." in s:
                out.append(s)
            else:
                out.append(f"{s}.NS")
        return out
    if country in ("us", "usa", "unitedstates", "united_states"):
        return [s.strip() for s in raw_symbols if s.strip()]
    raise ValueError("country must be one of: us, india")


# ----------------------------
# Strategy helpers
# ----------------------------
def barssince_last_true(condition: pd.Series) -> float:
    """
    Equivalent-ish to Pine's ta.barssince(condition) for the LAST bar only.
    Returns number of bars since condition was True.
    If condition was never True, returns np.nan.
    """
    if condition.empty:
        return np.nan
    true_idx = np.flatnonzero(condition.to_numpy(dtype=bool))
    if true_idx.size == 0:
        return np.nan
    return (len(condition) - 1) - true_idx[-1]


def screen_one_ticker(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    vol_lookback: int,
    vol_multiplier: float,
    recency: int,
    lower_close_ratio: float,
) -> tuple[bool, dict]:
    """
    df must have columns: Open, Close, Volume
    Returns (passes, debug_info)

    IMPORTANT DESIGN:
    - is_green uses RAW close/open (strict)
    - volume condition uses RAW volume + avg volume (strict)
    - recency uses STRICT "above both" = raw close > raw EMA
    - EMA "above both now" can be RELAXED:
        raw close > lower_close_ratio * EMA  (ratio < 1 makes it easier)
    """
    min_needed = max(ema_slow, vol_lookback) + 5
    if len(df) < min_needed:
        return False, {"reason": f"not enough data (have {len(df)}, need ~{min_needed})"}

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    avg_vol = vol.rolling(vol_lookback).mean()

    # STRICT aboveBoth for recency (uses raw close vs raw EMA)
    above_both_strict = (close > ema_f) & (close > ema_s)

    # RELAXED aboveBoth for the "above now" gate (allow close slightly below EMA)
    # Example: lower_close_ratio=0.97 => accept close > 0.97*EMA
    # This is LENIENT. (Your previous close*=ratio was stricter.)
    if lower_close_ratio <= 0:
        # Safety: nonsense ratios should not explode the screener
        lower_close_ratio = 1.0
    above_both_relaxed = (close > (ema_f * lower_close_ratio)) & (close > (ema_s * lower_close_ratio))

    # Fresh timing uses STRICT recency logic (exactly as you requested)
    # Pine intent:
    # barsSinceAbove = ta.barssince(not aboveBoth)
    # within = barsSinceAbove > 0 and barsSinceAbove <= recency
    bars_since_not_above = barssince_last_true(~above_both_strict)
    within_recency = (
        pd.notna(bars_since_not_above)
        and (bars_since_not_above > 0)
        and (bars_since_not_above <= recency)
    )

    # Latest candle checks (STRICT)
    is_green = close.iloc[-1] > open_.iloc[-1]
    vol_cond = (
        pd.notna(avg_vol.iloc[-1]) and (vol.iloc[-1] > (avg_vol.iloc[-1] * vol_multiplier))
    )
    above_now_relaxed = bool(above_both_relaxed.iloc[-1])

    buy = bool(is_green and vol_cond and above_now_relaxed and within_recency)

    debug = {
        "last_date": str(df.index[-1]),
        "latest_close": float(close.iloc[-1]),
        "latest_open": float(open_.iloc[-1]),
        "ema_fast": float(ema_f.iloc[-1]),
        "ema_slow": float(ema_s.iloc[-1]),
        "latest_vol": float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else np.nan,
        "avg_vol": float(avg_vol.iloc[-1]) if pd.notna(avg_vol.iloc[-1]) else np.nan,
        "bars_since_not_aboveBoth_strict": float(bars_since_not_above) if pd.notna(bars_since_not_above) else np.nan,
        "is_green": bool(is_green),
        "vol_cond": bool(vol_cond),
        "above_both_now_strict": bool(above_both_strict.iloc[-1]),
        "above_both_now_relaxed": bool(above_now_relaxed),
        "within_recency_strict": bool(within_recency),
        "lower_close_ratio": float(lower_close_ratio),
    }
    return buy, debug


# ----------------------------
# Market-time helpers (NEW)
# ----------------------------
def _market_clock(country: str) -> tuple[str, dtime, dtime]:
    """
    Returns (timezone_name, open_time, close_time) for regular session.
    Approx RTH times; does not account for holidays (fine for screener).
    """
    c = country.lower()
    if c in ("us", "usa", "unitedstates", "united_states"):
        return ("America/New_York", dtime(9, 30), dtime(16, 0))
    if c in ("india", "in"):
        return ("Asia/Kolkata", dtime(9, 15), dtime(15, 30))
    raise ValueError("country must be one of: us, india")


def _is_market_open_now(country: str, now_utc: datetime) -> tuple[bool, str]:
    """
    Returns (is_open, debug_string). Uses UTC -> exchange timezone conversion.
    Laptop timezone does NOT matter.
    """
    tz_name, open_t, close_t = _market_clock(country)
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)

    # Weekends closed
    if now_local.weekday() >= 5:
        return False, f"{tz_name} now={now_local} (weekend)"

    is_open = open_t <= now_local.time() < close_t
    return is_open, f"{tz_name} now={now_local} open={open_t} close={close_t} is_open={is_open}"


def drop_ongoing_daily_bar(df: pd.DataFrame, country: str, sym: str, now_utc: datetime, verbose: bool) -> pd.DataFrame:
    """
    If market is open now and the latest row matches today's exchange-local date,
    drop that last row (because it's the forming/ongoing session daily bar).
    Works even if df.index is tz-naive (yfinance typical daily output).
    """
    if df is None or df.empty:
        return df

    is_open, dbg = _is_market_open_now(country, now_utc)
    if verbose:
        print(f"[MARKET] {sym}: {dbg}")

    if not is_open:
        return df

    tz_name, _, _ = _market_clock(country)
    today_local = now_utc.astimezone(ZoneInfo(tz_name)).date()

    last_date = df.index[-1].date()
    if last_date == today_local:
        if verbose:
            print(f"[DROP TODAY BAR] {sym}: dropping forming daily bar dated {last_date} (market open)")
        return df.iloc[:-1]

    return df


# ----------------------------
# Download
# ----------------------------
def download_prices(symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """
    Downloads OHLCV per ticker. Returns dict[ticker] -> DataFrame with Open/Close/Volume columns.
    Uses yf.download(group_by='ticker') for efficiency.
    """
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        raise ValueError("No data returned from yfinance.")

    out: dict[str, pd.DataFrame] = {}

    # Single ticker => single-level columns
    if isinstance(data.columns, pd.Index) and "Open" in data.columns:
        sym = symbols[0]
        df = data.copy()
        df = df.dropna(subset=["Open", "Close", "Volume"], how="any")
        out[sym] = df
        return out

    # Multi-ticker => columns MultiIndex: (TICKER, Field)
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Unexpected yfinance format: columns are not MultiIndex for multi-ticker download.")

    for sym in symbols:
        if sym not in data.columns.get_level_values(0):
            continue
        cols_needed = [(sym, "Open"), (sym, "Close"), (sym, "Volume")]
        if not all(c in data.columns for c in cols_needed):
            continue
        df = data.loc[:, cols_needed].copy()
        df.columns = ["Open", "Close", "Volume"]
        df = df.dropna(subset=["Open", "Close", "Volume"], how="any")
        out[sym] = df

    return out


# ----------------------------
# Cache
# ----------------------------
def _utc_date_str(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d")


def _cache_root_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here / ".cache_yf" / "prices"


def _symbols_hash(symbols: list[str]) -> str:
    """
    Short hash of the actual subset so same-count but different tickers won't collide.
    """
    joined = "\n".join(symbols).encode("utf-8", errors="ignore")
    return hashlib.sha1(joined).hexdigest()[:10]


def _cache_prefix(
    utc_date: str,
    country: str,
    num_tickers: int,
    symbols_subset_hash: str,
) -> str:
    country = country.lower().strip()
    # Includes UTC date + num tickers (your original requirements) + subset hash (fixes collisions)
    return f"{utc_date}_country-{country}_n-{num_tickers:04d}_h-{symbols_subset_hash}"


def _cache_path_for_serial(cache_dir: Path, prefix: str, serial: int) -> Path:
    return cache_dir / f"{prefix}_s-{serial:02d}.pkl"


def _list_cache_files_for_day(cache_dir: Path, prefix: str) -> list[Path]:
    return sorted(cache_dir.glob(f"{prefix}_s-*.pkl"))


def _find_latest_cache_file_for_day(cache_dir: Path, prefix: str) -> Path | None:
    files = _list_cache_files_for_day(cache_dir, prefix)
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _next_serial_for_day(cache_dir: Path, prefix: str) -> int:
    files = _list_cache_files_for_day(cache_dir, prefix)
    if not files:
        return 0
    max_s = -1
    for p in files:
        try:
            part = p.stem.split("_s-")[-1]
            s = int(part)
            max_s = max(max_s, s)
        except Exception:
            continue
    return max_s + 1


def _load_cache_file(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "meta" not in obj or "price_map" not in obj:
        raise ValueError(f"Cache file '{path}' is not in expected format.")
    return obj


def _validate_cache_meta(meta: dict, expected_country: str, expected_num: int, expected_hash: str, allow_date_mismatch: bool) -> None:
    got_num = int(meta.get("num_tickers", -1))
    got_country = str(meta.get("country", "")).lower()
    got_date = str(meta.get("utc_date", ""))
    got_hash = str(meta.get("symbols_hash", ""))

    if got_num != expected_num:
        raise ValueError(f"Cache mismatch: file has num_tickers={got_num}, but this run expects {expected_num}.")
    if got_country != expected_country.lower():
        raise ValueError(f"Cache mismatch: file has country='{got_country}', but this run expects '{expected_country.lower()}'.")
    if got_hash != expected_hash:
        raise ValueError(f"Cache mismatch: file has symbols_hash='{got_hash}', but this run expects '{expected_hash}'.")
    if not allow_date_mismatch:
        today = _utc_date_str()
        if got_date != today:
            raise ValueError(f"Cache mismatch: file has utc_date={got_date}, but today (UTC) is {today}.")


def get_price_map_with_cache(
    *,
    symbols: list[str],
    start: datetime,
    end: datetime,
    country: str,
    years: int,
    cache_mode: str,
    cache_file: str | None,
    cache_serial: int | None,
) -> tuple[dict[str, pd.DataFrame], str]:
    cache_dir = _cache_root_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    utc_date = _utc_date_str()
    subset_hash = _symbols_hash(symbols)
    prefix = _cache_prefix(utc_date, country, len(symbols), subset_hash)

    # Explicit: --cache-file
    if cache_file is not None:
        chosen = Path(cache_file).expanduser().resolve()
        if not chosen.exists():
            raise FileNotFoundError(f"--cache-file not found: {chosen}")
        obj = _load_cache_file(chosen)
        _validate_cache_meta(
            obj["meta"],
            expected_country=country,
            expected_num=len(symbols),
            expected_hash=subset_hash,
            allow_date_mismatch=True,
        )
        return obj["price_map"], f"[CACHE] Using explicit file: {chosen.name}"

    # Explicit: --cache-serial
    if cache_serial is not None:
        chosen = _cache_path_for_serial(cache_dir, prefix, cache_serial)
        if not chosen.exists():
            raise FileNotFoundError(f"--cache-serial {cache_serial} not found for today's key: {chosen.name}")
        obj = _load_cache_file(chosen)
        _validate_cache_meta(
            obj["meta"],
            expected_country=country,
            expected_num=len(symbols),
            expected_hash=subset_hash,
            allow_date_mismatch=False,
        )
        return obj["price_map"], f"[CACHE] Using serial {cache_serial:02d}: {chosen.name}"

    latest = _find_latest_cache_file_for_day(cache_dir, prefix)

    if cache_mode == "auto":
        if latest is not None:
            obj = _load_cache_file(latest)
            _validate_cache_meta(
                obj["meta"],
                expected_country=country,
                expected_num=len(symbols),
                expected_hash=subset_hash,
                allow_date_mismatch=False,
            )
            return obj["price_map"], f"[CACHE] Using latest cache: {latest.name}"

        # No same-day cache -> download + write
        price_map = download_prices(symbols, start=start, end=end)
        serial = _next_serial_for_day(cache_dir, prefix)
        out_path = _cache_path_for_serial(cache_dir, prefix, serial)
        meta = {
            "created_utc_iso": datetime.now(timezone.utc).isoformat(),
            "utc_date": utc_date,
            "country": country.lower(),
            "num_tickers": len(symbols),
            "symbols_hash": subset_hash,
            "years": int(years),
            "start_utc_iso": (start.replace(tzinfo=timezone.utc).isoformat() if start.tzinfo is None else start.astimezone(timezone.utc).isoformat()),
            "end_utc_iso": (end.replace(tzinfo=timezone.utc).isoformat() if end.tzinfo is None else end.astimezone(timezone.utc).isoformat()),
            "note": "Keyed by UTC date + country + number of tickers + subset hash.",
        }
        with open(out_path, "wb") as f:
            pickle.dump({"meta": meta, "price_map": price_map}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return price_map, f"[CACHE] Downloaded fresh data → wrote: {out_path.name}"

    if cache_mode == "new":
        price_map = download_prices(symbols, start=start, end=end)
        serial = _next_serial_for_day(cache_dir, prefix)
        out_path = _cache_path_for_serial(cache_dir, prefix, serial)
        meta = {
            "created_utc_iso": datetime.now(timezone.utc).isoformat(),
            "utc_date": utc_date,
            "country": country.lower(),
            "num_tickers": len(symbols),
            "symbols_hash": subset_hash,
            "years": int(years),
            "start_utc_iso": (start.replace(tzinfo=timezone.utc).isoformat() if start.tzinfo is None else start.astimezone(timezone.utc).isoformat()),
            "end_utc_iso": (end.replace(tzinfo=timezone.utc).isoformat() if end.tzinfo is None else end.astimezone(timezone.utc).isoformat()),
            "note": "Forced new cache serial.",
        }
        with open(out_path, "wb") as f:
            pickle.dump({"meta": meta, "price_map": price_map}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return price_map, f"[CACHE] Forced new cache → wrote: {out_path.name}"

    raise ValueError(f"Unknown cache_mode: {cache_mode}")


# ----------------------------
# Earnings date printing (unchanged behavior: print only)
# ----------------------------
def safe_earnings_date_str(sym: str) -> str:
    try:
        cal = yf.Ticker(sym).calendar
        if not cal:
            return "N/A"
        ed = cal.get("Earnings Date")
        if isinstance(ed, list) and len(ed) > 0:
            return str(ed[0])
        return str(ed) if ed is not None else "N/A"
    except Exception:
        return "N/A"


def main():
    parser = argparse.ArgumentParser(description="EMA + Volume Strike Zone Screener (US + India)")
    parser.add_argument("--country", choices=["us", "india"], default="us",help="Market to screen")
    parser.add_argument("--csv", default="us_stocks.csv", help="CSV file containing symbols")
    parser.add_argument("--symbol-col", default="Symbol", help="Column name in CSV for symbols (default: Symbol)")

    # Universe selection
    parser.add_argument("--limit", type=int, default=None, help="Top N symbols from CSV (default: all unless rank range is used)")
    parser.add_argument("--rank-start", type=int, default=None, help="1-based inclusive start rank in CSV (e.g. 2500)")
    parser.add_argument("--rank-end", type=int, default=None, help="1-based inclusive end rank in CSV (e.g. 2700)")

    parser.add_argument("--years", type=int, default=2, help="Years of history to consider (default: 2)")

    # Friend's logic params
    parser.add_argument("--ema-fast", type=int, default=13, help="Fast EMA length (default: 13)")
    parser.add_argument("--ema-slow", type=int, default=21, help="Slow EMA length (default: 21)")
    parser.add_argument("--vol-mult", type=float, default=1.5, help="Volume multiplier (default: 1.5)")
    parser.add_argument("--vol-lookback", type=int, default=30, help="Volume SMA lookback (default: 30)")
    parser.add_argument("--recency", type=int, default=3, help="Max number of bars since breakout (default: 3)")
    parser.add_argument(
        "--lower-close-ratio",
        type=float,
        default=1.0,
        help="Relax EMA checks by allowing Close > (ratio * EMA). Example 0.97 means allow close slightly below EMA (default: 1.0 strict).",
    )

    # Cache flags
    parser.add_argument(
        "--cache-mode",
        choices=["auto", "new"],
        default="auto",
        help="Cache behavior: auto=use same-day cached data for (UTC date,country,num tickers,subset hash), new=force download and write new cache",
    )
    parser.add_argument("--cache-file", default=None, help="Use a specific cache file path (date mismatch allowed; must match country/num/subset)")
    parser.add_argument("--cache-serial", type=int, default=None, help="Use a specific serial cache for today's key (e.g., 0,1,2...)")

    parser.add_argument("--verbose", action="store_true", help="Print debug info for each passing ticker")
    args = parser.parse_args()

    raw = read_symbols_from_csv(args.csv, args.symbol_col)
    raw = slice_by_rank(raw, limit=args.limit, rank_start=args.rank_start, rank_end=args.rank_end)
    symbols = to_country_symbols(raw, args.country)

    if not symbols:
        print("No symbols selected (empty universe). Check --limit / --rank-start / --rank-end.")
        return

    end = datetime.utcnow()
    # Add buffer to ensure indicators compute cleanly
    start = end - timedelta(days=(args.years * 365) + 120)

    price_map, cache_msg = get_price_map_with_cache(
        symbols=symbols,
        start=start,
        end=end,
        country=args.country,
        years=args.years,
        cache_mode=args.cache_mode,
        cache_file=args.cache_file,
        cache_serial=args.cache_serial,
    )
    print(cache_msg)

    winners: list[str] = []

    # NEW: compute now_utc once, and use exchange-time logic to drop forming daily bar
    now_utc = datetime.now(timezone.utc)

    printed_date_debug = False
    for sym in symbols:
        df = price_map.get(sym)
        if df is None or df.empty:
            continue

        # Hard last-N-years filter (matches your intent)
        cutoff = end - timedelta(days=args.years * 365)
        df2 = df[df.index >= cutoff]
        if df2.empty:
            continue

        # NEW: drop ongoing daily bar if market is currently open (country-aware)
        df2 = drop_ongoing_daily_bar(df2, args.country, sym, now_utc, args.verbose)
        if df2.empty:
            continue

        # NEW: print which date we are using for "latest close" (for any ONE ticker)
        if not printed_date_debug:
            tz_name, _, _ = _market_clock(args.country)
            used_date = df2.index[-1].date()
            today_local = now_utc.astimezone(ZoneInfo(tz_name)).date()
            print(
                f"[DEBUG USED CLOSE DATE] {sym}: using last_date={used_date} "
                f"(exchange_tz={tz_name}, exchange_today={today_local}, utc_now={now_utc.isoformat()})"
            )
            printed_date_debug = True

        ok, dbg = screen_one_ticker(
            df2,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            vol_lookback=args.vol_lookback,
            vol_multiplier=args.vol_mult,
            recency=args.recency,
            lower_close_ratio=args.lower_close_ratio,
        )
        if ok:
            winners.append(sym)
            if args.verbose:
                print(f"\n{sym} PASSES")
                for k, v in dbg.items():
                    print(f"  {k}: {v}")

    print("\n=== RESULTS ===")
    print(f"Country: {args.country}")
    if args.rank_start is not None or args.rank_end is not None:
        print(f"CSV rank range: {args.rank_start or 1}..{args.rank_end or len(raw)} (1-based, inclusive)")
    if args.limit is not None and args.rank_start is None and args.rank_end is None:
        print(f"CSV top-N: {args.limit}")

    print(f"Screened: {len(symbols)} tickers (data returned for {len(price_map)})")
    print(f"Passed: {len(winners)}\n")

    for w in winners:
        print(f"{w:<24} EarningsDate={safe_earnings_date_str(w)}")


if __name__ == "__main__":
    main()