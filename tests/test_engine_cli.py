import json

from compaction_bench.cli import main
from compaction_bench.demo import demo_tasks
from compaction_bench.engine import render_json, render_markdown, run_bench


def test_run_bench_covers_all_strategies():
    report = run_bench(demo_tasks())
    assert len(report.aggregates) == 5
    assert all(a.tasks == 3 for a in report.aggregates)


def test_aggregates_sorted_by_success_then_cost():
    report = run_bench(demo_tasks())
    rates = [a.success_rate for a in report.aggregates]
    assert rates == sorted(rates, reverse=True)
    assert report.aggregates[0].success_rate == 1.0


def test_cost_estimate():
    report = run_bench(demo_tasks(), price_per_million=3.0)
    full = next(a for a in report.aggregates if a.strategy == "full")
    assert full.est_cost_usd is not None
    assert abs(full.est_cost_usd - full.total_context_tokens * 3.0 / 1_000_000) < 1e-9


def test_no_cost_without_price():
    report = run_bench(demo_tasks())
    assert all(a.est_cost_usd is None for a in report.aggregates)


def test_subset_of_strategies():
    report = run_bench(demo_tasks(), strategy_names=["full"])
    assert [a.strategy for a in report.aggregates] == ["full"]


def test_render_markdown_table():
    md = render_markdown(run_bench(demo_tasks()))
    assert md.startswith("| Strategy |")
    for name in ("full", "single-action", "end-of-conversation",
                 "sliding-window", "periodic-summary"):
        assert name in md


def test_render_json_parses():
    payload = json.loads(render_json(run_bench(demo_tasks(), price_per_million=1.0)))
    assert payload["price_per_million"] == 1.0
    assert len(payload["strategies"]) == 5
    assert len(payload["per_task"]) == 15


# --- CLI ------------------------------------------------------------------

def test_cli_demo_markdown(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "single-action" in out


def test_cli_demo_json(capsys):
    assert main(["demo", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["strategies"]) == 5


def test_cli_strategies(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "end-of-conversation" in out


def test_cli_run_spec(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"tasks": [
        {"name": "t", "steps": [{"tokens": 10, "critical": True}, {"tokens": 5}],
         "checkpoints": [{"at_step": 2, "requires": [0]}]},
    ]}), encoding="utf-8")
    assert main(["run", str(spec)]) == 0
    assert "full" in capsys.readouterr().out


def test_cli_run_bad_spec(tmp_path, capsys):
    spec = tmp_path / "bad.json"
    spec.write_text("{}", encoding="utf-8")
    assert main(["run", str(spec)]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_unknown_strategy(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"tasks": [
        {"name": "t", "steps": [{"tokens": 1}]},
    ]}), encoding="utf-8")
    assert main(["run", str(spec), "--strategies", "bogus"]) == 2


def test_cli_demo_with_params(capsys):
    assert main(["demo", "--summary-capacity", "32", "--price", "2.5"]) == 0
    out = capsys.readouterr().out
    assert "$" in out
    # with room for all 20 facts, single-action now passes fact-flood too
    assert "3/3 (100%)" in out
