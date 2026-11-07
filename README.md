# compaction-bench

**Compare agent context-compaction strategies on the same traces — success rate, peak context, and token cost, in one table.**

Long agent runs overflow context windows, so every serious harness compacts: it summarizes history and throws the original away. But *which* compaction strategy to use is usually chosen by gut feeling. Recent measurements (NVIDIA's "You don't need long context", 2026) show the choice matters enormously — aggressive per-step compaction matched full-history accuracy on SWE-bench Verified while cutting median peak context from 32K to 8K tokens, and a second compaction at the end of a conversation did even better. compaction-bench lets you run that comparison on *your* traces before you commit to a strategy.

## Quickstart

```bash
pip install -e .
compaction-bench demo
```

```
| Strategy | Success | Median peak ctx | Max peak ctx | Total ctx tokens | Est. cost |
|---|---|---|---|---|---|
| end-of-conversation | 3/3 (100%) | 11,600 | 23,400 | 688,800 | — |
| full | 3/3 (100%) | 11,600 | 23,400 | 780,000 | — |
| single-action | 2/3 (67%) | 700 | 800 | 64,100 | — |
| periodic-summary | 2/3 (67%) | 3,700 | 4,400 | 179,600 | — |
| sliding-window | 0/3 (0%) | 2,000 | 2,400 | 177,000 | — |
```

Add `--price 3.0` (USD per 1M input tokens) for a cost column, or `--format json` for machine-readable output.

## How it works

A bench spec describes recorded runs in the only terms a compaction strategy can act on:

- **steps** — each step's token size; a step can be marked `critical` (it establishes a fact the agent may need later) and tagged with a `segment` (which conversation in a chain it belongs to).
- **checkpoints** — the moments success is decided: "just before step 38, the facts from steps 2 and 17 must still be available."

```json
{
  "tasks": [
    {
      "name": "long-debug-session",
      "steps": [
        {"tokens": 600, "critical": true},
        {"tokens": 600}
      ],
      "checkpoints": [{"at_step": 38, "requires": [2, 17]}]
    }
  ]
}
```

```bash
compaction-bench run my-spec.json --price 3.0
```

The simulator replays each task under each strategy and reports, per strategy: success rate, median and max **peak context** (the largest prompt the model had to read), **total context tokens** (the run's input-token bill), and an optional cost estimate.

Summaries are *modelled*, not generated: a summary costs a fixed `--summary-tokens` (default 200) and retains up to `--summary-capacity` critical facts (default 16), evicting the oldest first. That FIFO eviction is deliberate — it reproduces the standard failure mode of a fixed-size running summary, which is exactly what the built-in `fact-flood` demo task demonstrates.

## Strategies

| Strategy | Behaviour |
|---|---|
| `full` | No compaction; the entire history stays in context. |
| `single-action` | After every step, older history folds into a running summary; context = summary + latest step. |
| `end-of-conversation` | Full history within a conversation; each finished conversation collapses into a summary that later conversations carry. |
| `sliding-window` | Keep only the last `--window` steps verbatim; older facts are gone. |
| `periodic-summary` | Every `--period` steps, fold all history into the summary. |

## Honest limits

This is a deterministic simulator, not a model runner. It answers "given these traces and these retention rules, which strategy keeps the needed facts at what token cost?" It does not answer "will a real summary preserve this fact?" — summary *quality* is compressed into two knobs (token budget, fact capacity) so strategies can be compared reproducibly and for free. Use it to shortlist strategies and size budgets, then validate the winner with real runs.

## Credits

The strategy set and the demo design are motivated by NVIDIA's paper **"You don't need long context"** (Belcak, Molchanov et al., 2026) — single-action and end-of-conversation compaction — as reported in the Latent Space AINews roundup of 2026-11-04: <http://latent.space/p/ainews-it-s-swapping-season/>. Related work on making compaction itself faster: SpecCompact (arXiv:2611.06876).

## Regenerate this project

This repository was designed to be reproducible from a single prompt. Paste the following into an LLM to generate a similar project:

> Write a small Python command-line tool, standard library only, that
> benchmarks context-compaction strategies for LLM agents on recorded
> traces, without calling a real model. A task is described as a
> sequence of steps with token sizes, where some steps are marked as
> establishing critical facts that later checkpoints depend on; each
> strategy (keeping the full history, keeping a running summary,
> summarizing at conversation boundaries, keeping a sliding window, or
> summarizing periodically) is simulated deterministically, with
> summaries costing a fixed token budget and holding only a bounded
> number of critical facts. The tool reports, per strategy, how many
> checkpoints still succeed plus peak and total context size and an
> optional cost estimate, so the trade-off between cheaper context and
> forgotten facts is visible side by side. Include a demo whose built-in
> tasks make the strategies visibly differ, including one where the
> summary overflows.

## License

MIT
