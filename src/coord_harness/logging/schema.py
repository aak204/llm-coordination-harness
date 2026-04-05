from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from coord_harness.core.enums import RegimeLabel


class RunSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    trial_id: str
    framework_id: str
    stage: str
    mode: str
    baseline: str
    enable_reasoning: bool = False
    attack_scenario: str | None = None
    attack_injection_depth: str | None = None
    seed: int
    status: str
    started_at: str
    completed_at: str
    config_digest: str
    decision_prompt_template_version: str
    message_serializer_version: str


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    panel_tier: str
    provider: str
    model_id: str
    requested_model_id: str
    allow_auto_routing: bool
    routing: dict
    temperature: float
    top_p: float
    max_completion_tokens: int
    strict_routing_required: bool
    runtime_metadata: dict = Field(default_factory=dict)
    route_metadata_examples: list[dict] = Field(default_factory=list)


class BenchmarkSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    stage: str
    split: str
    dataset_revision: str
    task_schema_version: str | None = None
    dataset_digest: str
    dataset_path: str
    manifest_path: str
    source_kind: str
    stubbed: bool
    task_count: int
    task_ids: list[str]
    scoring: str


class TopologySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str
    agent_count: int
    root_agent: str
    edge_count: int
    max_in_degree: int
    max_out_degree: int
    edges: list[tuple[str, str]]


class BudgetSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_billed_token_budget_per_task: int
    message_token_budget: int
    task_prompt_tokens: list[int]
    task_completion_tokens: list[int]
    task_billed_tokens: list[int]
    task_inter_agent_tokens: list[int]
    task_model_call_counts: list[int]
    mean_billed_tokens: float
    max_billed_tokens: int
    mean_inter_agent_tokens: float
    max_inter_agent_tokens: int


class OutcomeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_count: int
    accuracy: float
    score_mean: float
    score_std: float
    score_sum: float
    lift_vs_sa_star: float | None = None
    regime: RegimeLabel | None = None
    attacked_tasks: int = 0
    attack_success_rate: float | None = None
    infection_spread_rate: float | None = None
    quarantine_strength: float | None = None


class DerivedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    F: float | None = None
    rho: float | None = None
    B: float | None = None
    C: float | None = None
    B_status: str = "placeholder"
    recomputed_offline: bool = False
    F_source: str | None = None
    rho_source: str | None = None
    B_source: str | None = None
    C_source: str | None = None
    total_messages: int
    mean_message_tokens: float
    total_dropped_peer_tokens_raw: int = 0
    max_incoming_peer_tokens_raw: int = 0
    max_peer_context_quota: int = 0
    notes: list[str] = Field(default_factory=list)


class ProvenanceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str
    output_dir: str
    python_version: str
    platform: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    model_catalog_snapshot_path: str | None = None
    decision_prompt_hash: str
    message_serializer_hash: str
    model_catalog_snapshot_hash: str | None = None
    model_catalog_snapshot_fetched_at: str | None = None


class AttackAnalysisSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean_reference_experiment_id: str | None = None
    clean_reference_experiment_dir: str | None = None
    clean_reference_summary_path: str | None = None
    clean_F: float | None = None
    stress_F: float | None = None
    F_delta_vs_clean: float | None = None
    score_delta_vs_clean: float | None = None
    infection_spread_rate: float | None = None
    attack_success_rate: float | None = None
    quarantine_strength: float | None = None
    enable_reasoning: bool = False
    attack_scenario: str | None = None
    attack_injection_depth: str | None = None
    notes: list[str] = Field(default_factory=list)


class TrialSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunSection
    model: ModelSection
    benchmark: BenchmarkSection
    topology: TopologySection
    budget: BudgetSection
    outcomes: OutcomeSection
    derived: DerivedSection
    provenance: ProvenanceSection
    events_path: str
    attack_analysis: AttackAnalysisSection | None = None


class BatchIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: str
    summary_path: str
    benchmark_family: str
    topology_preset: str
    message_token_budget: int
    model_alias: str
    baseline: str
    enable_reasoning: bool = False
    attack_scenario: str | None = None
    seed: int
    score_mean: float
    accuracy: float
    lift_vs_sa_star: float | None = None
    regime: RegimeLabel | None = None


class BatchIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    generated_at: str
    trials: list[BatchIndexEntry]
