"""Built-in demo tasks, so `compaction-bench demo` works with no files.

The three tasks are shaped to make the strategies differ:

- ``long-debug-session``: one long conversation; two facts established
  early and mid-run are needed near the end. A short sliding window
  loses both; summaries keep them.
- ``research-chain``: three conversations in a chain; the last one
  needs a fact from each of the first two. End-of-conversation
  summaries carry them across.
- ``fact-flood``: more critical facts than a default summary can hold;
  the oldest fact is still required at the end, so fixed-capacity
  summaries evict it and fail, while full history passes.
"""

from __future__ import annotations

from .model import Checkpoint, Step, Task


def _steps(count: int, tokens: int, critical_at: set[int],
           segment: int = 0) -> tuple[Step, ...]:
    return tuple(
        Step(tokens=tokens, critical=i in critical_at, segment=segment)
        for i in range(count)
    )


def demo_tasks() -> list[Task]:
    debug = Task(
        name="long-debug-session",
        steps=_steps(40, 600, critical_at={2, 17}),
        checkpoints=(Checkpoint(at_step=38, requires=(2, 17)),),
    )

    chain_steps: list[Step] = []
    for seg in range(3):
        for i in range(8):
            idx = seg * 8 + i
            chain_steps.append(
                Step(tokens=500, critical=idx in (1, 11), segment=seg)
            )
    chain = Task(
        name="research-chain",
        steps=tuple(chain_steps),
        checkpoints=(Checkpoint(at_step=22, requires=(1, 11)),),
    )

    flood = Task(
        name="fact-flood",
        steps=_steps(30, 400, critical_at=set(range(20))),
        checkpoints=(Checkpoint(at_step=30, requires=(0,)),),
    )

    return [debug, chain, flood]
