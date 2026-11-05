import pytest

from compaction_bench.demo import demo_tasks
from compaction_bench.model import Checkpoint, Step, Task
from compaction_bench.strategies import (
    StrategyParams,
    run_strategy,
)

DEFAULTS = StrategyParams()


def _task(name, steps, checkpoints=()):
    return Task(name=name, steps=tuple(steps), checkpoints=tuple(checkpoints))


def _by_name(tasks, name):
    return next(t for t in tasks if t.name == name)


# --- full -----------------------------------------------------------------

def test_full_always_succeeds_and_peak_is_total():
    task = _task("t", [Step(100), Step(50, critical=True), Step(25)],
                 [Checkpoint(at_step=2, requires=(1,))])
    r = run_strategy("full", task, DEFAULTS)
    assert r.success
    assert r.peak_context_tokens == 150  # context before the last step
    assert r.total_context_tokens == 0 + 100 + 150


def test_full_end_checkpoint():
    task = _task("t", [Step(10, critical=True)],
                 [Checkpoint(at_step=1, requires=(0,))])
    assert run_strategy("full", task, DEFAULTS).success


# --- single-action --------------------------------------------------------

def test_single_action_peak_is_summary_plus_one_step():
    task = _by_name(demo_tasks(), "long-debug-session")
    r = run_strategy("single-action", task, DEFAULTS)
    assert r.success
    assert r.peak_context_tokens == 200 + 600


def test_single_action_fails_when_summary_capacity_exceeded():
    task = _by_name(demo_tasks(), "fact-flood")
    r = run_strategy("single-action", task, DEFAULTS)
    assert not r.success
    assert r.failed_checkpoints == [0]


def test_single_action_succeeds_with_bigger_capacity():
    task = _by_name(demo_tasks(), "fact-flood")
    params = StrategyParams(summary_capacity=32)
    assert run_strategy("single-action", task, params).success


def test_single_action_total_far_below_full():
    task = _by_name(demo_tasks(), "long-debug-session")
    full = run_strategy("full", task, DEFAULTS)
    single = run_strategy("single-action", task, DEFAULTS)
    assert single.total_context_tokens < full.total_context_tokens / 10


# --- sliding-window -------------------------------------------------------

def test_sliding_window_loses_old_fact():
    task = _by_name(demo_tasks(), "long-debug-session")
    r = run_strategy("sliding-window", task, DEFAULTS)
    assert not r.success


def test_sliding_window_keeps_recent_fact():
    task = _task("t", [Step(10), Step(10), Step(10, critical=True), Step(10)],
                 [Checkpoint(at_step=4, requires=(2,))])
    r = run_strategy("sliding-window", task, StrategyParams(window=4))
    assert r.success
    assert r.peak_context_tokens == 30


def test_sliding_window_zero_window_fails():
    task = _task("t", [Step(10, critical=True), Step(10)],
                 [Checkpoint(at_step=2, requires=(0,))])
    r = run_strategy("sliding-window", task, StrategyParams(window=0))
    assert not r.success


# --- end-of-conversation --------------------------------------------------

def test_eoc_carries_facts_across_segments():
    task = _by_name(demo_tasks(), "research-chain")
    r = run_strategy("end-of-conversation", task, DEFAULTS)
    assert r.success


def test_eoc_matches_full_within_one_segment():
    task = _by_name(demo_tasks(), "long-debug-session")
    eoc = run_strategy("end-of-conversation", task, DEFAULTS)
    full = run_strategy("full", task, DEFAULTS)
    assert eoc.success == full.success
    assert eoc.peak_context_tokens == full.peak_context_tokens


def test_eoc_peak_below_full_on_chain():
    task = _by_name(demo_tasks(), "research-chain")
    eoc = run_strategy("end-of-conversation", task, DEFAULTS)
    full = run_strategy("full", task, DEFAULTS)
    assert eoc.peak_context_tokens < full.peak_context_tokens


# --- periodic-summary -----------------------------------------------------

def test_periodic_summary_keeps_facts_within_capacity():
    task = _by_name(demo_tasks(), "long-debug-session")
    r = run_strategy("periodic-summary", task, DEFAULTS)
    assert r.success
    assert r.peak_context_tokens < 5000


def test_periodic_summary_fails_fact_flood():
    task = _by_name(demo_tasks(), "fact-flood")
    r = run_strategy("periodic-summary", task, DEFAULTS)
    assert not r.success


def test_periodic_summary_period_one_is_all_summary():
    task = _task("t", [Step(100), Step(100, critical=True), Step(100)],
                 [Checkpoint(at_step=3, requires=(1,))])
    r = run_strategy("periodic-summary", task, StrategyParams(period=1))
    assert r.success
    assert r.peak_context_tokens <= 200 + 100


# --- params / dispatch ----------------------------------------------------

def test_params_validation():
    with pytest.raises(ValueError):
        StrategyParams(period=0)
    with pytest.raises(ValueError):
        StrategyParams(window=-1)
    with pytest.raises(ValueError):
        StrategyParams(summary_capacity=-1)


def test_unknown_strategy_rejected():
    task = _task("t", [Step(1)])
    with pytest.raises(ValueError):
        run_strategy("nope", task, DEFAULTS)


def test_demo_outcome_matrix():
    tasks = demo_tasks()
    expected = {
        "full": {t.name: True for t in tasks},
        "sliding-window": {t.name: False for t in tasks},
    }
    for strategy, outcomes in expected.items():
        for task in tasks:
            r = run_strategy(strategy, task, DEFAULTS)
            assert r.success is outcomes[task.name], (strategy, task.name)
