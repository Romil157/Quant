#!/usr/bin/env python
"""Run a backtest with specified configuration."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", default="configs/backtest.yaml", help="Config file")
    parser.add_argument("--strategy", help="Override strategy name")
    parser.add_argument("--start", help="Override start date")
    parser.add_argument("--end", help="Override end date")
    parser.add_argument("--capital", type=float, help="Override initial capital")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.strategy:
        config["strategy"]["name"] = args.strategy
    if args.start:
        config["backtest"]["start"] = args.start
    if args.end:
        config["backtest"]["end"] = args.end
    if args.capital:
        config["backtest"]["initial_capital"] = args.capital
    
    print("Running backtest with config:")
    print(yaml.dump(config, default_flow_style=False))
    
    # TODO: Implement actual backtest engine
    print("\n[TODO] Backtest engine not yet implemented")
    print("This will be implemented in Phase 4")


if __name__ == "__main__":
    main()