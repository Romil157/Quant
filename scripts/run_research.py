#!/usr/bin/env python
"""Run research analysis with specified configuration."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run research analysis")
    parser.add_argument("--config", default="configs/research.yaml", help="Config file")
    parser.add_argument("--experiment", help="Experiment name")
    parser.add_argument("--symbols", nargs="+", help="Override universe")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.symbols:
        config["research"]["universe"] = args.symbols


    # TODO: Implement research engine


if __name__ == "__main__":
    main()
