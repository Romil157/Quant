#!/usr/bin/env python
"""Generate research report from backtest results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Generate research report")
    parser.add_argument("--results", required=True, help="Path to backtest results")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--format", choices=["html", "pdf", "markdown"], default="html")

    parser.parse_args()


    # TODO: Implement report generation


if __name__ == "__main__":
    main()
