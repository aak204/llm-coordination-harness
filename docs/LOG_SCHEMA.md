# Log Schema

Each trial writes:

- `summary.json`
- `events.jsonl`
- optional `model_catalog_snapshot.json`

`summary.json` contains these top-level sections:

- `run`: experiment and trial identity, clean/stress stage, `research_strict` vs `dev_convenience`, baseline, seed, timestamps, config digest, frozen prompt/serializer versions
- `model`: alias, panel tier, requested model ID, routing configuration, strict-routing flag, runtime metadata, route metadata examples
- `benchmark`: family, stage, split, dataset revision, dataset digest, dataset/manifest paths, task ids, scoring rule
- `topology`: preset, root, edge list, degree stats
- `budget`: per-task billed budget cap, message cap, billed token vectors, inter-agent token vectors, model call counts
- `outcomes`: accuracy, score moments, optional lift vs `sa_star`, optional regime
- `derived`: `F`, `rho`, `B`, `C`, offline recompute sources, dropped-token stats, notes on reliability
- `provenance`: config path, output path, Python/platform, optional git info, optional model catalog snapshot, prompt/serializer hashes, catalog snapshot hash/timestamp

`events.jsonl` stores timestamped task events such as:

- `agent_decision`
- `message_sent`
- `vote_aggregate`
- `final_aggregate`
- `fusion_skipped`

`agent_decision` events now log:

- `phase`: `single`, `local`, or `fusion`
- `visible_message_count`
- `visible_message_token_count`
- `local_state_snapshot`
- `peer_context_quota`
- `incoming_peer_tokens_raw`
- `incoming_peer_tokens_used`
- `dropped_peer_tokens_raw`
- `route_metadata`

Current operationalization for core variables:

- `F`: offline replay of sender local state vs delivered message snapshot answer fidelity
- `rho`: offline replay from paired `vote_local` no-communication baseline
- `B`: placeholder `1 - gini(messages_sent_per_agent)` with explicit placeholder status
- `C`: offline replay of `max(incoming_peer_tokens_raw / peer_context_quota)` with drop stats preserved in logs

OpenRouter runtime metadata logged when available:

- requested model ID
- requested provider policy
- resolved model ID
- resolved provider name/tag
- inferred fallback status
- pricing snapshot from model catalog and resolved endpoint
- model catalog snapshot hash and fetch timestamp
- endpoint snapshot hash and fetch timestamp

These are harness-level v1 operational definitions, not yet claim-locked scientific definitions.
