# Technical Report Draft

## Summary

This repository implements a reproducible clean-stage and stress-stage evaluation harness for multi-agent LLM systems under fixed billed-token budgets.

The central question is narrow:

Can a configuration-level predictor built from `F`, `rho`, `B`, `C` predict `help / saturation / collapse` better than a heuristic baseline that mostly sees benchmark family, topology, model identity, budget heuristics, and message-size statistics?

The current answer is:

- the measurement system works
- the coordination physics are visible
- the current held-out predictor does not yet pass the intended clean gate

That negative result is real and should not be patched over.

## What Is Measured

### Run Metadata

Each trial logs:

- experiment id
- trial id
- framework id
- stage
- mode
- baseline
- seed
- prompt template version
- serializer version

### Model Metadata

Each trial logs:

- requested model id
- requested provider policy
- resolved provider route
- resolved model id
- fallback inference
- pricing snapshot when available
- model catalog snapshot hash and timestamp when available

### Benchmark Metadata

Each trial logs:

- family
- split
- dataset revision
- task schema version
- dataset digest
- dataset path
- manifest path

### Topology Metadata

Each trial logs:

- topology preset
- node count
- root agent
- edge list
- degree statistics

### Budget Metadata

Each trial logs:

- billed prompt tokens
- billed completion tokens
- billed total tokens
- inter-agent tokens
- model call count

### Message-Level Events

`events.jsonl` contains:

- local decisions
- fusion decisions
- message sends
- fusion-skipped events
- optional attack-applied events in stress runs

The current event payloads include:

- sender local state snapshot
- receiver state before / after
- message snapshot
- peer-token quota
- incoming peer tokens raw
- dropped peer tokens raw

## Derived Variables

### F

Current definition:

`F = mean critical-fact survival per hop`

Operationally:

1. mark critical facts in task metadata
2. trace the path from the originating leaf toward the root
3. for each hop, check whether the sender-supported fact is still supported by the receiver state after fusion
4. average over all critical-fact hops

Current source label:

`offline_replay:critical_fact_survival_per_hop`

### rho

Current definition:

`rho = mean pairwise correlation of no-communication agent errors`

Operationally:

1. take the paired `vote_local` run for the same family / topology / budget / model / seed
2. extract per-agent correctness bits before any communication
3. compute pairwise binary correlations
4. average them

Current source label:

`offline_replay:paired_vote_local_no_communication_baseline`

### B

Current definition:

`B = 1 - gini(edge fact survival ratio)`

Operationally:

1. for each communication edge, measure how much critical-fact mass enters that edge
2. measure how much of that fact mass survives after the receiver update
3. compute the survival ratio for each edge
4. compute `1 - gini(ratios)`

Current source label:

`offline_replay:1_minus_gini(edge_fact_survival_ratio)`

This is no longer a placeholder-only slot, but it is still an operationalization rather than a final theoretically privileged invariant.

### C

Current definition:

`C = max(incoming_peer_tokens_raw / peer_context_quota)`

Operationally:

1. read fusion events
2. inspect raw incoming peer-token load and quota
3. take the maximum ratio over the trial

Current source label:

`offline_replay:max(incoming_peer_tokens_raw / peer_context_quota)`

## Independence From Final Score

The current extractors are not allowed to collapse into "just read the final answer and relabel it".

This was tested explicitly in [test_replay.py](d:/Dev/TEST_HYPO/tests/test_replay.py).

The synthetic counterexample encodes:

- final `score = 0`
- but a critical fact survives for part of the path

The test verifies that:

- `F = 0.666667`
- `B < 1.0`
- `F != score`

This is the key sanity check showing that the metrics describe graph physics rather than merely copying the final correctness bit.

## Predictor Analysis

An offline predictor analysis was run on the calibrated full clean batch:

- batch: [batch_index.json](d:/Dev/TEST_HYPO/outputs/p0a-calibrated-full-live/batch_index.json)
- gate: [gate_report.json](d:/Dev/TEST_HYPO/outputs/p0a-calibrated-full-live/gate_report.json)
- predictor analysis: [predictor_analysis.json](d:/Dev/TEST_HYPO/outputs/p0a-calibrated-full-live/predictor_analysis.json)

The analysis excludes `budget == 0` from the training split to avoid trivial collapse shortcuts.

It compares:

- heuristic-only feature set
- heuristic + `F/rho/B/C`

using a nonlinear local model (`RandomForestClassifier`) with held-out family evaluation.

## What The Predictor Is Actually Using

### Heuristic Model

For `help_vs_rest`, grouped feature importance is dominated by size proxies:

- `mean_billed_tokens`: `0.479423`
- `topology`: `0.117734`
- `model_alias`: `0.100229`
- `mean_message_tokens`: `0.092390`
- `mean_inter_agent_tokens`: `0.089920`

This is exactly the expected shortcut behavior of a cheating baseline.

### Core Model

For `help_vs_rest`, grouped feature importance shifts toward coordination physics:

- `rho`: `0.362065`
- `mean_billed_tokens`: `0.173410`
- `F`: `0.150415`
- `B`: `0.082979`
- `C`: `0.035493`

For `non_saturation_vs_rest`, the same pattern appears:

- `rho`: `0.381626`
- `F`: `0.157660`
- `mean_billed_tokens`: `0.147274`
- `B`: `0.093925`
- `C`: `0.025921`

This is the strongest positive result of the predictor analysis:

the core predictor is selecting physics-like variables instead of leaning primarily on token-count shortcuts.

## Held-Out Results

The clean gate is still not passed.

### help_vs_rest

Held-out family, nonzero-budget:

- `agentsnet_mini`: heuristic RF `0.571429`, core RF `0.607143`, delta `+0.035714`
- `craft_mini`: heuristic RF `0.459259`, core RF `0.488889`, delta `+0.029630`

### collapse_vs_rest

After excluding `budget == 0` from train, collapse largely disappears as a learnable signal.

Both families show:

- `insufficient_train_label_variation`

This is not a bug. It means the previously strong collapse signal was heavily driven by trivial zero-budget cases.

### non_saturation_vs_rest

Held-out family, nonzero-budget:

- `agentsnet_mini`: heuristic RF `0.571429`, core RF `0.607143`, delta `+0.035714`
- `craft_mini`: heuristic RF `0.414815`, core RF `0.488889`, delta `+0.074074`

The core model improves over heuristic in some slices, but not enough to satisfy the clean gate.

## Mechanistic Evidence

Even though the gate fails, the coordination physics are visible.

Examples:

- `AgentsNet-mini`, `gemini`, `MA-FT`, `9 agents`
  - `star`, `budget 96`: `score=1.0`, `F=1.0`, `B=1.0`
  - `balanced_tree`, `budget 96`: `score=0.75`, `F=0.916667`, `B=0.95`

- `CRAFT-mini`, `gemini`, `MA-FT`, `9 agents`
  - `star`, `budget 96`: `score=1.0`, `F=1.0`, `B=1.0`
  - `balanced_tree`, `budget 96`: `score=0.75`, `F=0.833333`, `B=0.95`

In both cases, topology-sensitive score degradation becomes visible in `F` and `B`, while `C` moves only slightly.

## P0b Attack Layer

An attack layer was added in stress mode with one compromised leaf agent that is forced to emit a malicious wrong answer and propagate that answer outward.

Artifacts:

- attack batch: [batch_index.json](d:/Dev/TEST_HYPO/outputs/p0b-attacks-live/batch_index.json)

Current stress metrics:

- `attack_success_rate`
- `infection_spread_rate`

The most careful phrasing is:

P0b shows measurable adversarial propagation, but current results do not support a monotonic topology-vulnerability claim. In some slices, structures that degrade useful coordination may also weakly attenuate malicious propagation.

This is mechanistically interesting, but it is not yet strong enough to support a clean attack-robustness claim.

## Release Figures

- [feature_importance_help_vs_rest.png](d:/Dev/TEST_HYPO/docs/figures/feature_importance_help_vs_rest.png)
- [topology_penalty_budget96_maft.png](d:/Dev/TEST_HYPO/docs/figures/topology_penalty_budget96_maft.png)
- [topology_delta_budget96_maft.png](d:/Dev/TEST_HYPO/docs/figures/topology_delta_budget96_maft.png)
- [predictor_holdout_auroc_nonzero.png](d:/Dev/TEST_HYPO/docs/figures/predictor_holdout_auroc_nonzero.png)
- [attack_score_delta_vs_clean.png](d:/Dev/TEST_HYPO/docs/figures/attack_score_delta_vs_clean.png)
- [attack_infection_spread.png](d:/Dev/TEST_HYPO/docs/figures/attack_infection_spread.png)
- [attack_success_rate.png](d:/Dev/TEST_HYPO/docs/figures/attack_success_rate.png)

## Limitations

### Lack of Organic Collapse At Nonzero Budgets

This is the central limitation of the current v1 dataset.

After removing trivial `budget == 0` shortcuts:

- collapse becomes sparse
- held-out collapse prediction stops being a meaningful supervised problem

This is not a harness bug. It is an empirical property of current strong LLMs on the calibrated benchmark families.

### Predictor Failure Is Real

The current core predictor does not pass the clean gate.

That result should be reported honestly.

The correct interpretation is:

- the measurement rig works
- the coordination variables are not arbitrary
- but the current v1 predictor is still too weak to deliver the intended held-out classification performance

### B Is Better, But Not Final

The current `B` is already meaningful and reacts to topology-sensitive failures.

However, it is still an operationalization rather than a final theoretically privileged definition.

It should be treated as a strong v1 extractor, not as a fully matured order parameter.

## Bottom Line

This repository has already achieved something scientifically valuable:

1. it extracts coordination variables from logs rather than from hidden in-memory state
2. those variables are demonstrably not copies of final score
3. `F` and `B` respond to topology-induced coordination failures in both benchmark families
4. heuristic baselines do in fact rely heavily on token-size shortcuts
5. the current held-out predictor still fails the clean gate
6. stress runs show measurable adversarial propagation, but not a simple topology-vulnerability ordering

That is a legitimate negative result, and it should be preserved as such.
