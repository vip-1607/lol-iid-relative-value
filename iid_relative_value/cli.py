"""Command line interface for i.i.d. relative-value analysis."""

import argparse
import json
from pathlib import Path

from .analysis import analyze_relative_value, result_as_dict
from .model import Market, SeriesState
from .portfolio import FeeModel, Quote


def _market(data):
    return Market(data["kind"], data.get("line"))


def _quote(data):
    fee = data.get("fees", {})
    return Quote(_market(data["market"]), data["side"].upper(), data.get("bid", 0.0),
                 data["ask"], data["depth"], FeeModel(**fee), data.get("slippage", 0.0))


def run(config):
    state = SeriesState(config["best_of"], config.get("score", {}).get("a", 0),
                        config.get("score", {}).get("b", 0))
    anchor = config["anchor"]
    quotes = {q["name"]: _quote(q) for q in config["quotes"]}
    return analyze_relative_value(anchor["bid"], anchor["ask"], _market(anchor["market"]),
                                  quotes, state, config.get("bankroll", 1.0),
                                  config.get("edge_buffer", 0.0),
                                  config.get("fractional_kelly", 0.25))


def _table(result):
    print(f"status={result.status}  p={result.anchor.p}  p_intervals={result.anchor.p_intervals}")
    print(f"structural={result.structural_status}  worst_pnl={result.structural_worst_pnl:.6f}")
    print(f"{'name':24} {'side':4} {'fair':>9} {'cost':>9} {'worst edge':>12} status")
    for row in result.markets:
        fair = "-" if row.fair_point is None else f"{row.fair_point:.6f}"
        edge = "-" if row.worst_edge is None else f"{row.worst_edge:.6f}"
        print(f"{row.name[:24]:24} {row.side:4} {fair:>9} {row.unit_cost:9.6f} {edge:>12} {row.status}")
    if result.kelly:
        print("fractional Kelly stakes:", result.kelly.stakes)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="iid-rv")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("config", type=Path)
    analyze.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    if args.format == "json":
        print(json.dumps(result_as_dict(result), ensure_ascii=False, indent=2))
    else:
        _table(result)


if __name__ == "__main__":
    main()
