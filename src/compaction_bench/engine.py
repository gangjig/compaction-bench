"""Run a bench (tasks x strategies) and render the comparison."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field

from .model import Task
from .strategies import STRATEGIES, RunResult, StrategyParams, run_strategy


@dataclass
class StrategyAggregate:
    strategy: str
    tasks: int
    successes: int
    success_rate: float
    median_peak_context_tokens: float
    max_peak_context_tokens: int
    total_context_tokens: int
    est_cost_usd: float | None


@dataclass
class BenchReport:
    results: list[RunResult]
    aggregates: list[StrategyAggregate]
    price_per_million: float | None = None
    per_task: list[dict] = field(default_factory=list)


def run_bench(
    tasks: list[Task],
    strategy_names: list[str] | None = None,
    params: StrategyParams | None = None,
    price_per_million: float | None = None,
) -> BenchReport:
    params = params or StrategyParams()
    names = strategy_names or sorted(STRATEGIES)
    results: list[RunResult] = []
    for name in names:
        for task in tasks:
            results.append(run_strategy(name, task, params))

    aggregates: list[StrategyAggregate] = []
    for name in names:
        rs = [r for r in results if r.strategy == name]
        peaks = [r.peak_context_tokens for r in rs]
        total_tokens = sum(r.total_context_tokens for r in rs)
        aggregates.append(
            StrategyAggregate(
                strategy=name,
                tasks=len(rs),
                successes=sum(1 for r in rs if r.success),
                success_rate=(sum(1 for r in rs if r.success) / len(rs)) if rs else 0.0,
                median_peak_context_tokens=statistics.median(peaks) if peaks else 0,
                max_peak_context_tokens=max(peaks) if peaks else 0,
                total_context_tokens=total_tokens,
                est_cost_usd=(
                    total_tokens * price_per_million / 1_000_000
                    if price_per_million is not None
                    else None
                ),
            )
        )
    # Rank: success rate first, then cheapest total tokens.
    aggregates.sort(key=lambda a: (-a.success_rate, a.total_context_tokens))

    per_task = [
        {
            "task": r.task,
            "strategy": r.strategy,
            "success": r.success,
            "peak_context_tokens": r.peak_context_tokens,
            "total_context_tokens": r.total_context_tokens,
            "failed_checkpoints": r.failed_checkpoints,
        }
        for r in results
    ]
    return BenchReport(
        results=results,
        aggregates=aggregates,
        price_per_million=price_per_million,
        per_task=per_task,
    )


def render_markdown(report: BenchReport) -> str:
    lines = [
        "| Strategy | Success | Median peak ctx | Max peak ctx | Total ctx tokens | Est. cost |",
        "|---|---|---|---|---|---|",
    ]
    for a in report.aggregates:
        cost = f"${a.est_cost_usd:,.2f}" if a.est_cost_usd is not None else "—"
        median_peak = (
            f"{a.median_peak_context_tokens:,.0f}"
            if float(a.median_peak_context_tokens).is_integer()
            else f"{a.median_peak_context_tokens:,.1f}"
        )
        lines.append(
            f"| {a.strategy} | {a.successes}/{a.tasks} "
            f"({a.success_rate:.0%}) | {median_peak} | "
            f"{a.max_peak_context_tokens:,} | {a.total_context_tokens:,} | {cost} |"
        )
    return "\n".join(lines) + "\n"


def render_json(report: BenchReport) -> str:
    payload = {
        "price_per_million": report.price_per_million,
        "strategies": [asdict(a) for a in report.aggregates],
        "per_task": report.per_task,
    }
    return json.dumps(payload, indent=2) + "\n"
