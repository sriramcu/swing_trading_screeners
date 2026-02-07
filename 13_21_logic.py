#!/usr/bin/env python3
"""
EMA + Volume Strike Zone Screener (US + India)

Screens tickers for the following "buy condition" on the latest candle:
- Green candle: Close > Open
- Volume spike: Volume > SMA(Volume, lookback) * vol_multiplier
- Above both EMAs: Close > EMA(fast) AND Close > EMA(slow)
- "Within 3 bars of breakout": barssince(not aboveBoth) is 1..3
- Data is limited to last N years (handled by download range + filter)

No trading, no backtesting, no paper trading. Screening only.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone

# [new]
import pickle
# [new]
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def read_symbols_from_csv(csv_path: str, symbol_col: str = "Symbol") -> list[str]:
    symbols = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if symbol_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain a '{symbol_col}' column. Found: {reader.fieldnames}")
        for row in reader:
            sym = (row.get(symbol_col) or "").strip()
            if sym:
                symbols.append(sym)
    return symbols


def to_country_symbols(raw_symbols: list[str], country: str) -> list[str]:
    country = country.lower()
    if country in ("india", "in"):
        # Append .NS if not already a suffix like .NS / .BO etc.
        out = []
        for s in raw_symbols:
            if "." in s:
                out.append(s)
            else:
                out.append(f"{s}.NS")
        return out
    elif country in ("us", "usa", "unitedstates", "united_states"):
        return raw_symbols
    else:
        raise ValueError("country must be one of: us, india")


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
) -> tuple[bool, dict]:
    """
    df must have columns: Open, Close, Volume
    Returns (passes, debug_info)
    """
    # Require enough rows for indicators
    min_needed = max(ema_slow, vol_lookback) + 5
    if len(df) < min_needed:
        return False, {"reason": f"not enough data (have {len(df)}, need ~{min_needed})"}

    close = df["Close"]
    open_ = df["Open"]
    vol = df["Volume"]

    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    avg_vol = vol.rolling(vol_lookback).mean()

    above_both = (close > ema_f) & (close > ema_s)

    # Pine:
    # barsSinceAbove = ta.barssince(not aboveBoth)
    # withinThreeBars = barsSinceAbove > 0 and barsSinceAbove <= 3
    bars_since_not_above = barssince_last_true(~above_both)
    within_three = pd.notna(bars_since_not_above) and (bars_since_not_above > 0) and (bars_since_not_above <= 3)

    is_green = close.iloc[-1] > open_.iloc[-1]
    vol_cond = vol.iloc[-1] > (avg_vol.iloc[-1] * vol_multiplier) if pd.notna(avg_vol.iloc[-1]) else False
    above_now = bool(above_both.iloc[-1])

    buy = is_green and vol_cond and above_now and within_three

    debug = {
        "latest_close": float(close.iloc[-1]),
        "latest_open": float(open_.iloc[-1]),
        "ema_fast": float(ema_f.iloc[-1]),
        "ema_slow": float(ema_s.iloc[-1]),
        "latest_vol": float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else np.nan,
        "avg_vol": float(avg_vol.iloc[-1]) if pd.notna(avg_vol.iloc[-1]) else np.nan,
        "bars_since_not_aboveBoth": float(bars_since_not_above) if pd.notna(bars_since_not_above) else np.nan,
        "is_green": bool(is_green),
        "vol_cond": bool(vol_cond),
        "above_both_now": bool(above_now),
        "within_three_bars_of_breakout": bool(within_three),
    }
    return buy, debug


def download_prices(symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """
    Downloads OHLCV per ticker. Returns dict[ticker] -> DataFrame with Open/Close/Volume columns.
    Uses yf.download(group_by='ticker') so multi-ticker download is efficient.
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

    # If only one ticker, yfinance returns single-level columns like ["Open","High",...]
    if isinstance(data.columns, pd.Index) and "Open" in data.columns:
        sym = symbols[0]
        df = data.copy()
        df = df.dropna(subset=["Open", "Close", "Volume"], how="any")
        out[sym] = df
        return out

    # Multi-ticker: columns are like (TICKER, 'Open'), (TICKER, 'Close'), ...
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


# [new]
def _utc_date_str(dt: datetime | None = None) -> str:
    """Returns UTC date as YYYYMMDD."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d")


# [new]
def _cache_root_dir() -> Path:
    """
    Cache lives in a subfolder next to this script (based on __file__), not the current working directory.
    """
    here = Path(__file__).resolve().parent
    return here / ".cache_yf" / "prices"


# [new]
def _cache_prefix(utc_date: str, country: str, num_tickers: int) -> str:
    """
    Keyed by UTC date + country + number of tickers (per your requirement).
    Name must include UTC date and number of tickers.
    """
    country = country.lower().strip()
    return f"{utc_date}_country-{country}_n-{num_tickers:04d}"


# [new]
def _cache_path_for_serial(cache_dir: Path, prefix: str, serial: int) -> Path:
    return cache_dir / f"{prefix}_s-{serial:02d}.pkl"


# [new]
def _list_cache_files_for_day(cache_dir: Path, prefix: str) -> list[Path]:
    return sorted(cache_dir.glob(f"{prefix}_s-*.pkl"))


# [new]
def _find_latest_cache_file_for_day(cache_dir: Path, prefix: str) -> Path | None:
    files = _list_cache_files_for_day(cache_dir, prefix)
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


# [new]
def _next_serial_for_day(cache_dir: Path, prefix: str) -> int:
    files = _list_cache_files_for_day(cache_dir, prefix)
    if not files:
        return 0
    max_s = -1
    for p in files:
        try:
            part = p.stem.split("_s-")[-1]
            s = int(part)
            if s > max_s:
                max_s = s
        except Exception:
            continue
    return max_s + 1


# [new]
def _load_cache_file(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "meta" not in obj or "price_map" not in obj:
        raise ValueError(f"Cache file '{path}' is not in expected format.")
    return obj


# [new]
def _validate_cache_meta(meta: dict, expected_country: str, expected_num: int, allow_date_mismatch: bool) -> None:
    """
    Requirements:
    - mismatched num stocks -> error
    - mismatched date -> allowed if allow_date_mismatch=True
    Also enforce country match (safety).
    """
    got_num = int(meta.get("num_tickers", -1))
    got_country = str(meta.get("country", "")).lower()
    got_date = str(meta.get("utc_date", ""))

    if got_num != expected_num:
        raise ValueError(f"Cache mismatch: file has num_tickers={got_num}, but this run expects {expected_num}.")
    if got_country != expected_country.lower():
        raise ValueError(
            f"Cache mismatch: file has country='{got_country}', but this run expects '{expected_country.lower()}'."
        )
    if not allow_date_mismatch:
        today = _utc_date_str()
        if got_date != today:
            raise ValueError(f"Cache mismatch: file has utc_date={got_date}, but today (UTC) is {today}.")


# [modified]
# def get_price_map_with_cache(...)-> dict[str, pd.DataFrame]:
#     ...
# [modified]
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
    """
    cache_mode:
      - auto (default): use same-day cache if present else download and create cache
      - new: force download and create a new serial-suffixed cache for the day/key

    f2 behavior:
      - If f2 flags not mentioned at all: use most recent cache filename by default (same day/key) in auto mode
      - If --cache-file provided: use that file (date mismatch allowed; num mismatch NOT allowed)
      - If --cache-serial provided: pick that serial for today's key
    """
    cache_dir = _cache_root_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    utc_date = _utc_date_str()
    prefix = _cache_prefix(utc_date, country, len(symbols))

    # f2 explicit selection: --cache-file
    if cache_file is not None:
        chosen = Path(cache_file).expanduser().resolve()
        if not chosen.exists():
            raise FileNotFoundError(f"--cache-file not found: {chosen}")
        obj = _load_cache_file(chosen)
        _validate_cache_meta(obj["meta"], expected_country=country, expected_num=len(symbols), allow_date_mismatch=True)
        return obj["price_map"], f"[CACHE] Using explicit file: {chosen.name}"

    # f2 explicit selection: --cache-serial
    if cache_serial is not None:
        chosen = _cache_path_for_serial(cache_dir, prefix, cache_serial)
        if not chosen.exists():
            raise FileNotFoundError(f"--cache-serial {cache_serial} not found for today's key: {chosen.name}")
        obj = _load_cache_file(chosen)
        _validate_cache_meta(obj["meta"], expected_country=country, expected_num=len(symbols), allow_date_mismatch=True)
        return obj["price_map"], f"[CACHE] Using serial {cache_serial:02d}: {chosen.name}"

    # No f2 flags mentioned: default behavior uses "most recent cache filename" (mtime) for same day/key.
    latest = _find_latest_cache_file_for_day(cache_dir, prefix)

    if cache_mode == "auto":
        if latest is not None:
            obj = _load_cache_file(latest)
            _validate_cache_meta(obj["meta"], expected_country=country, expected_num=len(symbols), allow_date_mismatch=False)
            return obj["price_map"], f"[CACHE] Using latest cache: {latest.name}"

        # No same-day cache => download and create new serial
        price_map = download_prices(symbols, start=start, end=end)
        serial = _next_serial_for_day(cache_dir, prefix)
        out_path = _cache_path_for_serial(cache_dir, prefix, serial)
        meta = {
            "created_utc_iso": datetime.now(timezone.utc).isoformat(),
            "utc_date": utc_date,
            "country": country.lower(),
            "num_tickers": len(symbols),
            "years": int(years),
            "start_utc_iso": (
                start.replace(tzinfo=timezone.utc).isoformat()
                if start.tzinfo is None
                else start.astimezone(timezone.utc).isoformat()
            ),
            "end_utc_iso": (
                end.replace(tzinfo=timezone.utc).isoformat()
                if end.tzinfo is None
                else end.astimezone(timezone.utc).isoformat()
            ),
            "note": "Keyed by UTC date + country + number of tickers (per user request).",
        }
        with open(out_path, "wb") as f:
            pickle.dump({"meta": meta, "price_map": price_map}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return price_map, f"[CACHE] Downloaded fresh data → wrote: {out_path.name}"

    if cache_mode == "new":
        # Force download and store with next serial (even if cache exists)
        price_map = download_prices(symbols, start=start, end=end)
        serial = _next_serial_for_day(cache_dir, prefix)
        out_path = _cache_path_for_serial(cache_dir, prefix, serial)
        meta = {
            "created_utc_iso": datetime.now(timezone.utc).isoformat(),
            "utc_date": utc_date,
            "country": country.lower(),
            "num_tickers": len(symbols),
            "years": int(years),
            "start_utc_iso": (
                start.replace(tzinfo=timezone.utc).isoformat()
                if start.tzinfo is None
                else start.astimezone(timezone.utc).isoformat()
            ),
            "end_utc_iso": (
                end.replace(tzinfo=timezone.utc).isoformat()
                if end.tzinfo is None
                else end.astimezone(timezone.utc).isoformat()
            ),
            "note": "Forced new cache serial.",
        }
        with open(out_path, "wb") as f:
            pickle.dump({"meta": meta, "price_map": price_map}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return price_map, f"[CACHE] Forced new cache → wrote: {out_path.name}"

    raise ValueError(f"Unknown cache_mode: {cache_mode}")


def main():
    parser = argparse.ArgumentParser(description="EMA + Volume Strike Zone Screener (US + India)")
    parser.add_argument("--country", choices=["us", "india"], required=True, help="Market to screen")
    parser.add_argument("--csv", required=True, help="CSV file containing symbols")
    parser.add_argument("--symbol-col", default="Symbol", help="Column name in CSV for symbols (default: Symbol)")
    parser.add_argument("--limit", type=int, default=300, help="Max symbols to screen (default: 300)")
    parser.add_argument("--years", type=int, default=2, help="Years of history to consider (default: 2)")

    # Friend's logic params
    parser.add_argument("--ema-fast", type=int, default=13, help="Fast EMA length (default: 13)")
    parser.add_argument("--ema-slow", type=int, default=21, help="Slow EMA length (default: 21)")
    parser.add_argument("--vol-mult", type=float, default=1.5, help="Volume multiplier (default: 1.5)")
    parser.add_argument("--vol-lookback", type=int, default=30, help="Volume SMA lookback (default: 30)")

    # [new] cache flags
    parser.add_argument(
        "--cache-mode",
        choices=["auto", "new"],
        default="auto",
        help="Cache behavior: auto=use same-day cached data for (UTC date,country,num tickers), new=force download and write a new serial cache",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Use a specific cache file path (date mismatch allowed; num tickers must match; country must match)",
    )
    parser.add_argument(
        "--cache-serial",
        type=int,
        default=None,
        help="Use a specific serial cache for today's key (e.g., 0,1,2...). Only applies to today's (UTC date,country,num tickers).",
    )

    parser.add_argument("--verbose", action="store_true", help="Print debug info for each passing ticker")
    args = parser.parse_args()

    raw_symbols = read_symbols_from_csv(args.csv, args.symbol_col)
    raw_symbols = raw_symbols[: max(args.limit, 0)]
    symbols = to_country_symbols(raw_symbols, args.country)

    # Download enough buffer for indicators + "within 3 bars" logic.
    # Also: using calendar days; yfinance will return trading days only.
    end = datetime.utcnow()
    start = end - timedelta(days=(args.years * 365) + 120)

    # [modified]
    # price_map = download_prices(symbols, start=start, end=end)
    # [modified] (now returns (price_map, cache_msg))
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
    # [new]
    print(cache_msg)

    winners: list[str] = []
    for sym in symbols:
        df = price_map.get(sym)
        if df is None or df.empty:
            print(f"Failed to fetch data for {sym}")
            continue

        # Apply a hard "last N years" filter to match your friend's intent.
        cutoff = end - timedelta(days=args.years * 365)
        df2 = df[df.index >= cutoff]
        if df2.empty:
            continue

        ok, dbg = screen_one_ticker(
            df2,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            vol_lookback=args.vol_lookback,
            vol_multiplier=args.vol_mult,
        )
        if ok:
            winners.append(sym)
            if args.verbose:
                print(f"\n{sym} PASSES")
                for k, v in dbg.items():
                    print(f"  {k}: {v}")

    print("\n=== RESULTS ===")
    print(f"Country: {args.country}")
    print(f"Screened: {len(symbols)} tickers (data returned for {len(price_map)})")
    print(f"Passed: {len(winners)}\n")
    for w in winners:
        try:
            print(f"{w:<24} {yf.Ticker(w).calendar.get('Earnings Date')[0].strftime('%d-%m-%Y')}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"{w:<24} Unable to get earnings date")


if __name__ == "__main__":
    main()
