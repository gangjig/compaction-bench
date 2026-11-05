# Changelog

## 0.1.0 — 2026-11-05

Initial release.

- Bench spec model: tasks as step lists (token size, `critical` fact flags,
  conversation `segment` ids) plus checkpoints that name the facts which
  must still be available at a given step.
- Five strategies: `full`, `single-action`, `end-of-conversation`,
  `sliding-window`, `periodic-summary`. Summaries are modelled with a
  fixed token cost and a bounded fact capacity (FIFO eviction).
- Deterministic simulator reporting success, peak context, and total
  context tokens per run; aggregation ranked by success rate, then cost.
- CLI: `run`, `demo` (three built-in tasks), `strategies`; Markdown and
  JSON output; optional per-million-token price for cost estimates.
- Motivated by NVIDIA's "You don't need long context" (2026); see the
  Credits section of the README.
