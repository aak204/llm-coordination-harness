# Benchmark Spec

## Scope

P0a clean harness currently exposes:

- `craft_mini`
- `agentsnet_mini`

Both are repo-local clean benchmark packs backed by:

- `data/benchmarks/clean/craft_mini/`
- `data/benchmarks/clean/agentsnet_mini/`

Each family has:

- `manifest.yaml`
- `tasks.jsonl`

This is no longer a hardcoded stub adapter path. The harness now loads benchmark tasks from disk, computes dataset digests, and logs manifest provenance.

## Baselines

- `sa_star`: single-agent baseline under the same per-task billed token budget
- `vote_local`: no communication, local votes only
- `ma_ft`: topology-aware upward fusion tree over local decisions

## Topologies

- `star`
- `balanced_tree`
- `sparse_graph`

## Budget Presets

- `0`
- `32`
- `96`

`message_token_budget` controls per-message content cap. `total_billed_token_budget` remains fixed per task and tracks prompt + completion tokens.

`ma_ft` now performs actual message-conditioned fusion calls. For `message_token_budget = 0`, no inter-agent content is sent, no message events are logged, and `F` is left undefined.

## Clean vs Stress

Current repo implements `clean` only. Stress/attack runs are intentionally absent from the config schema and benchmark packs.

## Known Limitation

The current benchmark packs are repo-local mini datasets for first-pilot reproducibility, not an upstream benchmark release sync. Replacing them with a larger or externally maintained pack is now a data operation rather than a harness rewrite.
