#!/usr/bin/env python
"""Download market data from various providers.

Saves raw parquet for the parquet provider; for the mock provider the data is
generated on the fly. Reports per-symbol success/failure and exits non-zero if
any symbol failed.

Examples
--------
python scripts/download_data.py --symbols AAPL MSFT --start 2023-01-01 --end 2023-12-31
python scripts/download_data.py --symbols AAPL --provider yfinance --start 2023-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provider", default="mock", help="Data provider (mock/parquet/yfinance)")
    parser.add_argument("--output", default="data/raw", help="Output directory for parquet provider")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")

    args = parser.parse_args()

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: invalid date format (use YYYY-MM-DD): {e}", file=sys.stderr)
        sys.exit(2)

    if start >= end:
        print("Error: --start must be earlier than --end", file=sys.stderr)
        sys.exit(2)

    # Use the platform helper so the mock provider accepts the requested
    # symbols and the parquet provider persists to the right location.
    from quant.data import download_data as _helper

    failures: list[tuple[str, str]] = []
    total_bars = 0

    try:
        data = _helper(symbols=args.symbols, start_date=args.start, end_date=args.end,
                       provider=args.provider, data_root=Path(args.output))
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        sys.exit(1)

    for symbol, df in data.items():
        try:
            if df is None or len(df) == 0:
                failures.append((symbol, "no data returned"))
                print(f"  [fail] {symbol}: no data returned")
                continue
            bars = len(df)
            total_bars += bars
            print(f"  [ok]   {symbol}: {bars} bars  range={df.index.min()} -> {df.index.max()}")
        except Exception as e:
            failures.append((symbol, str(e)))
            print(f"  [fail] {symbol}: {e}")

    print()
    print(f"Summary: {len(args.symbols) - len(failures)}/{len(args.symbols)} symbols ok, "
          f"{total_bars} total bars")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
