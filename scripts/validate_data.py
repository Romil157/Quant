#!/usr/bin/env python
"""Validate market data quality."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant.data.providers.factory import ProviderFactory
from quant.data.validation import (
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


def main():
    parser = argparse.ArgumentParser(description="Validate market data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to validate")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provider", default="mock", help="Data provider")
    parser.add_argument("--data-root", default="data/raw", help="Data root directory")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")
    parser.add_argument("--freq", default="B", help="Expected frequency")

    args = parser.parse_args()

    from datetime import datetime
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    if args.provider == "mock":
        provider = ProviderFactory.create_provider(args.provider)
    else:
        provider = ProviderFactory.create_provider(
            args.provider, data_root=Path(args.data_root)
        )

    all_valid = True

    for symbol in args.symbols:

        try:
            data = provider.get_historical_data(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=args.timeframe,
            )


            # Run all validations
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

            for _name, validator in validators:
                result = validator(data)
                if result.warnings:
                    pass
                if not result.is_valid:
                    all_valid = False

        except Exception:
            all_valid = False

    if all_valid:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
