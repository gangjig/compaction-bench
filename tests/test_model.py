import json

import pytest

from compaction_bench.model import SpecError, Step, Task, load_spec, task_from_dict


def _spec(tmp_path, payload):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_task_from_dict_minimal():
    task = task_from_dict({"name": "t", "steps": [{"tokens": 10}]})
    assert task.name == "t"
    assert task.steps[0].tokens == 10
    assert task.steps[0].critical is False
    assert task.checkpoints == ()


def test_task_from_dict_full():
    task = task_from_dict({
        "name": "t",
        "steps": [
            {"tokens": 5, "critical": True, "segment": 0, "label": "fact"},
            {"tokens": 7},
        ],
        "checkpoints": [{"at_step": 1, "requires": [0]}],
    })
    assert task.steps[0].label == "fact"
    assert task.checkpoints[0].requires == (0,)


def test_negative_tokens_rejected():
    with pytest.raises(SpecError):
        task_from_dict({"name": "t", "steps": [{"tokens": -1}]})


def test_empty_steps_rejected():
    with pytest.raises(SpecError):
        task_from_dict({"name": "t", "steps": []})


def test_missing_name_rejected():
    with pytest.raises(SpecError):
        task_from_dict({"steps": [{"tokens": 1}]})


def test_checkpoint_requires_noncritical_rejected():
    with pytest.raises(SpecError):
        task_from_dict({
            "name": "t",
            "steps": [{"tokens": 1}, {"tokens": 1}],
            "checkpoints": [{"at_step": 1, "requires": [0]}],
        })


def test_checkpoint_requires_future_step_rejected():
    with pytest.raises(SpecError):
        task_from_dict({
            "name": "t",
            "steps": [{"tokens": 1, "critical": True}, {"tokens": 1}],
            "checkpoints": [{"at_step": 0, "requires": [0]}],
        })


def test_decreasing_segments_rejected():
    with pytest.raises(SpecError):
        Task(name="t", steps=(Step(1, segment=1), Step(1, segment=0)))


def test_load_spec_roundtrip(tmp_path):
    path = _spec(tmp_path, {"tasks": [
        {"name": "a", "steps": [{"tokens": 3}]},
        {"name": "b", "steps": [{"tokens": 4}]},
    ]})
    tasks = load_spec(path)
    assert [t.name for t in tasks] == ["a", "b"]


def test_load_spec_duplicate_names_rejected(tmp_path):
    path = _spec(tmp_path, {"tasks": [
        {"name": "a", "steps": [{"tokens": 3}]},
        {"name": "a", "steps": [{"tokens": 4}]},
    ]})
    with pytest.raises(SpecError):
        load_spec(path)


def test_load_spec_invalid_json_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec(p)


def test_load_spec_missing_tasks_rejected(tmp_path):
    path = _spec(tmp_path, {"nope": []})
    with pytest.raises(SpecError):
        load_spec(path)
