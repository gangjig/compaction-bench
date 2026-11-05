"""Command line interface for compaction-bench."""

from __future__ import annotations

import argparse
import sys

from .demo import demo_tasks
from .engine import render_json, render_markdown, run_bench
from .model import SpecError, load_spec
from .strategies import STRATEGIES, STRATEGY_DESCRIPTIONS, StrategyParams


def _add_param_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--summary-tokens", type=int, default=200,
                        help="token cost of a summary (default: 200)")
    parser.add_argument("--summary-capacity", type=int, default=16,
                        help="max critical facts a summary retains (default: 16)")
    parser.add_argument("--window", type=int, default=4,
                        help="sliding-window size in steps (default: 4)")
    parser.add_argument("--period", type=int, default=8,
                        help="periodic-summary fold interval in steps (default: 8)")
    parser.add_argument("--price", type=float, default=None,
                        help="USD per 1M input tokens, for the cost column")
    parser.add_argument("--strategies", default=None,
                        help="comma-separated subset of strategies to run")


def _params_from(args: argparse.Namespace) -> StrategyParams:
    return StrategyParams(
        summary_tokens=args.summary_tokens,
        summary_capacity=args.summary_capacity,
        window=args.window,
        period=args.period,
    )


def _strategy_names(args: argparse.Namespace) -> list[str] | None:
    if not args.strategies:
        return None
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in names if s not in STRATEGIES]
    if unknown:
        raise SpecError(
            f"unknown strategies: {unknown}; choose from {sorted(STRATEGIES)}"
        )
    return names


def _emit(args: argparse.Namespace, tasks) -> int:
    report = run_bench(
        tasks,
        strategy_names=_strategy_names(args),
        params=_params_from(args),
        price_per_million=args.price,
    )
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compaction-bench",
        description="Compare agent context-compaction strategies on the "
                    "same traces: success rate, peak context, token cost.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a bench spec JSON file")
    p_run.add_argument("spec", help="path to the spec JSON file")
    p_run.add_argument("--format", choices=["md", "json"], default="md")
    _add_param_args(p_run)

    p_demo = sub.add_parser("demo", help="run the built-in demo tasks")
    p_demo.add_argument("--format", choices=["md", "json"], default="md")
    _add_param_args(p_demo)

    sub.add_parser("strategies", help="list the available strategies")

    args = parser.parse_args(argv)
    try:
        if args.command == "strategies":
            for name in sorted(STRATEGIES):
                print(f"{name}: {STRATEGY_DESCRIPTIONS[name]}")
            return 0
        if args.command == "demo":
            return _emit(args, demo_tasks())
        tasks = load_spec(args.spec)
        return _emit(args, tasks)
    except (SpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
