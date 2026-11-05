"""Data model for compaction-bench.

A *task* is one recorded agent run, reduced to the facts a compaction
strategy can act on:

- ``steps``: the run's steps in order. Each step has a token size, an
  optional ``critical`` flag (the step establishes a fact the agent may
  need later), and an optional ``segment`` id (which conversation in a
  multi-conversation chain the step belongs to; used by the
  end-of-conversation strategy).
- ``checkpoints``: moments where success is decided. A checkpoint sits
  *before* a step (``at_step``) and lists the step indexes whose facts
  must still be available in context at that moment. A run succeeds iff
  every checkpoint's required facts are available.

This is a deterministic simulation model: availability is computed from
the strategy's retention rules, not by calling a model. That keeps runs
reproducible and free — the point is comparing strategies, not models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class SpecError(ValueError):
    """Raised when a bench spec is malformed."""


@dataclass(frozen=True)
class Step:
    tokens: int
    critical: bool = False
    segment: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise SpecError(f"step tokens must be >= 0, got {self.tokens}")
        if self.segment < 0:
            raise SpecError(f"step segment must be >= 0, got {self.segment}")


@dataclass(frozen=True)
class Checkpoint:
    at_step: int
    requires: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.at_step < 0:
            raise SpecError(f"checkpoint at_step must be >= 0, got {self.at_step}")
        for idx in self.requires:
            if idx < 0:
                raise SpecError(f"checkpoint requires negative step index {idx}")
            if idx >= self.at_step:
                raise SpecError(
                    f"checkpoint at step {self.at_step} requires step {idx}, "
                    "which has not happened yet"
                )


@dataclass(frozen=True)
class Task:
    name: str
    steps: tuple[Step, ...]
    checkpoints: tuple[Checkpoint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("task name must not be empty")
        if not self.steps:
            raise SpecError(f"task {self.name!r} has no steps")
        for cp in self.checkpoints:
            if cp.at_step > len(self.steps):
                raise SpecError(
                    f"task {self.name!r}: checkpoint at_step {cp.at_step} "
                    f"is past the end ({len(self.steps)} steps)"
                )
            for idx in cp.requires:
                if idx >= len(self.steps):
                    raise SpecError(
                        f"task {self.name!r}: checkpoint requires step {idx}, "
                        f"but the task has {len(self.steps)} steps"
                    )
                if not self.steps[idx].critical:
                    raise SpecError(
                        f"task {self.name!r}: checkpoint requires step {idx}, "
                        "which is not marked critical"
                    )
        segs = [s.segment for s in self.steps]
        if segs != sorted(segs):
            raise SpecError(
                f"task {self.name!r}: step segments must be non-decreasing"
            )


def task_from_dict(raw: object, where: str = "task") -> Task:
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: expected an object, got {type(raw).__name__}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise SpecError(f"{where}: 'name' must be a non-empty string")
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SpecError(f"task {name!r}: 'steps' must be a non-empty list")
    steps: list[Step] = []
    for i, rs in enumerate(raw_steps):
        if not isinstance(rs, dict):
            raise SpecError(f"task {name!r} step {i}: expected an object")
        tokens = rs.get("tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool):
            raise SpecError(f"task {name!r} step {i}: 'tokens' must be an integer")
        steps.append(
            Step(
                tokens=tokens,
                critical=bool(rs.get("critical", False)),
                segment=int(rs.get("segment", 0)),
                label=str(rs.get("label", "")),
            )
        )
    checkpoints: list[Checkpoint] = []
    for j, rc in enumerate(raw.get("checkpoints", [])):
        if not isinstance(rc, dict):
            raise SpecError(f"task {name!r} checkpoint {j}: expected an object")
        at = rc.get("at_step")
        req = rc.get("requires")
        if not isinstance(at, int) or isinstance(at, bool):
            raise SpecError(
                f"task {name!r} checkpoint {j}: 'at_step' must be an integer"
            )
        if not isinstance(req, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in req
        ):
            raise SpecError(
                f"task {name!r} checkpoint {j}: 'requires' must be a list of integers"
            )
        checkpoints.append(Checkpoint(at_step=at, requires=tuple(req)))
    return Task(name=name, steps=tuple(steps), checkpoints=tuple(checkpoints))


def load_spec(path: str | Path) -> list[Task]:
    """Load a bench spec JSON file: ``{"tasks": [task, ...]}``."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        raise SpecError(f"{path}: expected an object with a 'tasks' list")
    if not raw["tasks"]:
        raise SpecError(f"{path}: 'tasks' list is empty")
    tasks = [task_from_dict(t, where=f"tasks[{i}]") for i, t in enumerate(raw["tasks"])]
    names = [t.name for t in tasks]
    if len(names) != len(set(names)):
        raise SpecError(f"{path}: duplicate task names: {names}")
    return tasks
