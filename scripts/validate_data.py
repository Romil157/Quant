#!/usr/bin/env python
"""Validate market data quality and print a human-readable report.

Exit code is 0 only if every symbol passes every validator with no errors.

Examples
--------
python scripts/validate_data.py --symbols AAPL MSFT --start 2023-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from quant.data.providers.factory import ProviderFactory
from quant.data.validation import (
    ValidationResult,
    validate_data_gaps,
    validate_extreme_outliers,
    validate_no_duplicate_timestamps,
    validate_no_missing_timestamps,
    validate_no_negative_prices,
    validate_no_negative_volume,
    validate_no_zero_prices,
    validate_ohlc_relationships,
    validate_timezone_consistency,
    validate_trading_session_consistency,
)


def _print_result(name: str, result: ValidationResult) -> bool:
    """Print one validator's outcome; return True if valid."""
    status = "PASS" if result.is_valid else "FAIL"
    print(f"    {status:4}  {name}  warnings={len(result.warnings)}  issues={len(result.errors)}")
    for w in result.warnings[:3]:
        print(f"           warn: {w}")
    for e in result.errors[:3]:
        print(f"           err : {e}")
    return result.is_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate market data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to validate")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provider", default="mock", help="Data provider")
    parser.add_argument("--data-root", default="data/raw", help="Data root directory")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")
    parser.add_argument("--freq", default="B", help="Expected frequency")
    args = parser.parse_args()

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: invalid date format (use YYYY-MM-DD): {e}", file=sys.stderr)
        sys.exit(2)

    if args.provider == "mock":
        provider = ProviderFactory.create_provider(args.provider, symbols=args.symbols)
    else:
        provider = ProviderFactory.create_provider(
            args.provider, data_root=Path(args.data_root)
        )

    all_valid = True
    failures: list[str] = []

    for symbol in args.symbols:
        print(f"Validating {symbol}...")
        try:
            data = provider.get_historical_data(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=args.timeframe,
            )
        except Exception as e:
            print(f"    [fail] could not load data: {e}")
            all_valid = False
            failures.append(symbol)
            continue

        validators = [
            ("OHLC Relationships", validate_ohlc_relationships),
            ("Negative Prices", validate_no_negative_prices),
            ("Negative Volume", validate_no_negative_volume),
            ("Zero Prices", validate_no_zero_prices),
            ("Duplicate Timestamps", validate_no_duplicate_timestamps),
            ("Missing Timestamps", lambda d: validate_no_missing_timestamps(d, args.freq)),
            ("Timezone Consistency", validate_timezone_consistency),
            ("Extreme Outliers", validate_extreme_outliers),
            ("Data Gaps", lambda d: validate_data_gaps(d, args.timeframe)),
            ("Trading Session", validate_trading_session_consistency),
        ]

        symbol_ok = True
        for name, validator in validators:
            try:
                result = validator(data)
                if not _print_result(name, result):
                    symbol_ok = False
            except Exception as e:
                print(f"    ERR   {name}: validator crashed: {e}")
                symbol_ok = False

        if not symbol_ok:
            all_valid = False
            failures.append(symbol)
        print(f"  {symbol}: {'PASS' if symbol_ok else 'FAIL'}  ({len(data)} bars)")
        print()

    print("=" * 60)
    if all_valid:
        print(f"All {len(args.symbols)} symbols valid")
        sys.exit(0)
    else:
        succeeded = len(args.symbols) - len(failures)
        print(f"{succeeded}/{len(args.symbols)} symbols valid. Failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
