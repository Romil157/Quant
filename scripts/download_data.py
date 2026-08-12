#!/usr/bin/env python
"""Download market data from various providers."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant.data.providers.factory import ProviderFactory


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--provider", default="mock", help="Data provider")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")
    
    args = parser.parse_args()
    
    if args.provider == "mock":
        provider = ProviderFactory.create_provider(args.provider)
    else:
        provider = ProviderFactory.create_provider(args.provider, data_root=Path(args.output))
    
    from datetime import datetime
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    for symbol in args.symbols:
        print(f"Downloading {symbol} from {args.start} to {args.end}...")
        try:
            data = provider.get_historical_data(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=args.timeframe,
            )
            print(f"  Got {len(data)} bars for {symbol}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()