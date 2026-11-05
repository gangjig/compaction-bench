"""Compaction strategies and the deterministic run simulator.

A strategy decides, at every step of a task, what the agent's context
contains: which earlier steps are still present verbatim, and whether a
summary stands in for the rest. The simulator replays a task under a
strategy and reports:

- ``success``: every checkpoint's required facts were available;
- ``peak_context_tokens``: the largest context the model had to read;
- ``total_context_tokens``: the sum of context sizes over all steps —
  the token-bill proxy (later tokens cost the same per token on most
  APIs, so this tracks the input bill of the run).

Summaries are modelled, not generated: a summary costs a fixed
``summary_tokens`` and retains up to ``summary_capacity`` critical
facts, evicting the oldest first (FIFO) — the standard failure mode of
a fixed-size running summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Task


@dataclass(frozen=True)
class StrategyParams:
    summary_tokens: int = 200
    summary_capacity: int = 16
    window: int = 4
    period: int = 8

    def __post_init__(self) -> None:
        if self.summary_tokens < 0:
            raise ValueError("summary_tokens must be >= 0")
        if self.summary_capacity < 0:
            raise ValueError("summary_capacity must be >= 0")
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if self.period < 1:
            raise ValueError("period must be >= 1")


@dataclass
class RunResult:
    task: str
    strategy: str
    success: bool
    peak_context_tokens: int
    total_context_tokens: int
    failed_checkpoints: list[int] = field(default_factory=list)


class _Summary:
    """A fixed-budget summary of compacted steps (FIFO fact retention)."""

    def __init__(self, params: StrategyParams) -> None:
        self._params = params
        self.facts: list[int] = []  # critical step indexes, oldest first
        self.active = False

    def absorb(self, step_index: int, critical: bool) -> None:
        self.active = True
        if critical:
            self.facts.append(step_index)
            overflow = len(self.facts) - self._params.summary_capacity
            if overflow > 0:
                del self.facts[:overflow]

    @property
    def tokens(self) -> int:
        return self._params.summary_tokens if self.active else 0

    def covers(self, step_index: int) -> bool:
        return step_index in self.facts


def _checkpoints_at(task: Task) -> dict[int, list[int]]:
    """Map step index -> checkpoint positions evaluated at that point.

    Position ``i`` means "just before step i runs"; position
    ``len(steps)`` means "after the last step", against the final
    context state.
    """
    by_step: dict[int, list[int]] = {}
    for pos, cp in enumerate(task.checkpoints):
        by_step.setdefault(cp.at_step, []).append(pos)
    return by_step


def _evaluate(task: Task, by_step: dict[int, list[int]], at: int,
              available: set[int], failed: list[int]) -> None:
    for pos in by_step.get(at, []):
        cp = task.checkpoints[pos]
        if any(req not in available for req in cp.requires):
            failed.append(pos)


def _simulate_simple(task: Task, name: str, context_at) -> RunResult:
    """Simulator for strategies whose context is a pure function of the
    step index (no evolving state): context_at(i) returns
    (context_tokens, available_fact_indexes) just before step i runs,
    and context_at(len(steps)) returns the end state."""
    by_step = _checkpoints_at(task)
    peak = 0
    total = 0
    failed: list[int] = []
    n = len(task.steps)
    for i in range(n + 1):
        ctx, available = context_at(i)
        if i < n:
            peak = max(peak, ctx)
            total += ctx
        _evaluate(task, by_step, i, available, failed)
    return RunResult(task.name, name, not failed, peak, total, failed)


def run_full(task: Task, params: StrategyParams) -> RunResult:
    """No compaction: the whole history stays in context."""

    def context_at(i: int):
        ctx = sum(s.tokens for s in task.steps[:i])
        available = {j for j in range(i) if task.steps[j].critical}
        return ctx, available

    return _simulate_simple(task, "full", context_at)


def run_sliding_window(task: Task, params: StrategyParams) -> RunResult:
    """Keep only the last ``window`` steps verbatim; older facts are gone."""

    def context_at(i: int):
        start = max(0, i - params.window)
        ctx = sum(s.tokens for s in task.steps[start:i])
        available = {j for j in range(start, i) if task.steps[j].critical}
        return ctx, available

    return _simulate_simple(task, "sliding-window", context_at)


def run_single_action(task: Task, params: StrategyParams) -> RunResult:
    """One action per context: after every step, the older history folds
    into a running summary; the model sees the summary plus the single
    latest step verbatim.

    Mirrors the single-action compaction measured in NVIDIA's
    "You don't need long context" (2026): peak context stays near the
    summary size plus one step.
    """
    summary = _Summary(params)
    by_step = _checkpoints_at(task)
    peak = 0
    total = 0
    failed: list[int] = []
    n = len(task.steps)
    for i in range(n + 1):
        # Before step i runs, step i-2 and older are in the summary and
        # step i-1 is present verbatim.
        if i >= 2:
            summary.absorb(i - 2, task.steps[i - 2].critical)
        ctx = summary.tokens
        available = set(summary.facts)
        if 0 < i <= n:
            ctx += task.steps[i - 1].tokens
            if task.steps[i - 1].critical:
                available.add(i - 1)
        if i < n:
            peak = max(peak, ctx)
            total += ctx
        _evaluate(task, by_step, i, available, failed)
    return RunResult(task.name, "single-action", not failed, peak, total, failed)


def run_end_of_conversation(task: Task, params: StrategyParams) -> RunResult:
    """Full history inside a conversation; when a conversation ends, it
    collapses into a summary that later conversations carry with them."""
    # A segment's summary is fixed once the segment ends, so pre-build all.
    summaries: dict[int, _Summary] = {}
    for j, step in enumerate(task.steps):
        sm = summaries.setdefault(step.segment, _Summary(params))
        sm.absorb(j, step.critical)

    def context_at(i: int):
        if i == 0:
            return 0, set()
        current = task.steps[min(i, len(task.steps) - 1)].segment
        if i == len(task.steps):
            current = task.steps[-1].segment
        ctx = 0
        available: set[int] = set()
        for seg, sm in summaries.items():
            if seg < current:
                ctx += sm.tokens
                available |= set(sm.facts)
        for j in range(i):
            if task.steps[j].segment == current:
                ctx += task.steps[j].tokens
                if task.steps[j].critical:
                    available.add(j)
        return ctx, available

    return _simulate_simple(task, "end-of-conversation", context_at)


def run_periodic_summary(task: Task, params: StrategyParams) -> RunResult:
    """Every ``period`` steps, fold all history so far into the summary;
    between folds, context is the summary plus the steps since the fold."""
    summary = _Summary(params)
    by_step = _checkpoints_at(task)
    peak = 0
    total = 0
    failed: list[int] = []
    n = len(task.steps)
    folded_up_to = 0  # steps [0, folded_up_to) are inside the summary
    for i in range(n + 1):
        if i > 0 and i % params.period == 0:
            for j in range(folded_up_to, i):
                summary.absorb(j, task.steps[j].critical)
            folded_up_to = i
        ctx = summary.tokens + sum(
            s.tokens for s in task.steps[folded_up_to:i]
        )
        available = set(summary.facts)
        available |= {
            j for j in range(folded_up_to, i) if task.steps[j].critical
        }
        if i < n:
            peak = max(peak, ctx)
            total += ctx
        _evaluate(task, by_step, i, available, failed)
    return RunResult(task.name, "periodic-summary", not failed, peak, total, failed)


STRATEGIES = {
    "full": run_full,
    "single-action": run_single_action,
    "end-of-conversation": run_end_of_conversation,
    "sliding-window": run_sliding_window,
    "periodic-summary": run_periodic_summary,
}

STRATEGY_DESCRIPTIONS = {
    "full": "No compaction: keep the entire history in context.",
    "single-action": "After every step, fold older history into a running summary; context = summary + latest step.",
    "end-of-conversation": "Full history within a conversation; collapse each finished conversation into a summary.",
    "sliding-window": "Keep only the last N steps verbatim (param: window).",
    "periodic-summary": "Every K steps, fold all history into the summary (param: period).",
}


def run_strategy(name: str, task: Task, params: StrategyParams) -> RunResult:
    try:
        fn = STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}"
        ) from None
    return fn(task, params)
