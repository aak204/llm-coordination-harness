# Claim

This harness targets one narrow P0a clean claim and now implements the minimum mechanics required to run a first held-out family pilot.

At fixed total billed token budget and fixed orchestration setup, a multi-agent configuration can be characterized by four logged variables:

- `F`: message decision-fidelity
- `rho`: shared-error correlation
- `B`: propagation balance
- `C`: fan-in pressure

The v1 goal is to test whether `{F, rho, B, C}` predicts `help / saturation / collapse` better than a heuristic baseline that sees only benchmark family, topology, model identity, budget heuristics, and basic message-size statistics.

Current operational scope:

- clean stage only
- homogeneous teams only
- baselines: `sa_star`, `vote_local`, `ma_ft`
- benchmark families: `craft_mini`, `agentsnet_mini`
- topology presets: `star`, `balanced_tree`, `sparse_graph`
- message budgets: `0`, `32`, `96`
- fixed total billed token budget per task
- held-out family gate reporting with heuristic-vs-core predictor comparison

Out of scope for this harness version:

- attacks
- semantic acts
- auto-routing by default
- UI and dashboards
- protocol standardization
- paper prose beyond runbook-level documentation
