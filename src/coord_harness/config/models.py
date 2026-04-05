from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coord_harness.core.enums import (
    AttackInjectionDepth,
    AttackMode,
    AttackPayloadType,
    BaselineStrategy,
    BenchmarkFamily,
    ModelProvider,
    PanelTier,
    RunMode,
    RunStage,
    TopologyPreset,
)


P0A_MESSAGE_TOKEN_PRESETS = {0, 32, 96}


class ProviderRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: list[str] = Field(default_factory=list)
    only: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    allow_fallbacks: bool = False
    require_parameters: bool = False


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    effort: str | None = None
    max_tokens: int | None = None
    exclude: bool = True


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    provider: ModelProvider
    model_id: str
    panel_tier: PanelTier = PanelTier.SWEEP
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 128
    allow_auto_routing: bool = False
    routing: ProviderRoutingConfig = Field(default_factory=ProviderRoutingConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_auto_routing_consistency(self) -> "ModelSpec":
        if not self.allow_auto_routing and self.provider is ModelProvider.OPENROUTER:
            self.routing.allow_fallbacks = False
        return self


class BenchmarkAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: BenchmarkFamily
    stage: RunStage = RunStage.CLEAN
    split: str = "smoke"
    task_limit: int | None = None
    dataset_revision: str | None = None
    dataset_dir: Path | None = None
    task_schema_version: str | None = None
    stubbed: bool = False


class RegimeThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    help_min_lift: float = 0.02
    collapse_max_lift: float = -0.02


class AttackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    injection_depth: AttackInjectionDepth | None = None
    payload_type: AttackPayloadType = AttackPayloadType.HARD_HALLUCINATION
    target_answer_policy: str = "first_wrong_option"
    compromise_confidence: float = 0.99
    propagate_instruction: bool = True
    only_baselines: list[BaselineStrategy] = Field(default_factory=list)
    clean_reference_experiment_dir: Path | None = None
    attack_mode: AttackMode | None = None
    attacker_policy: str | None = None

    @field_validator("compromise_confidence")
    @classmethod
    def validate_compromise_confidence(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("compromise_confidence must be between 0.0 and 1.0")
        return value

    @model_validator(mode="after")
    def normalize_legacy_attack_fields(self) -> "AttackConfig":
        inferred_depth = self.injection_depth
        if inferred_depth is None:
            if self.attack_mode is AttackMode.COMPROMISED_LEAF or self.attacker_policy == "deepest_leaf":
                inferred_depth = AttackInjectionDepth.LEAF
            elif self.attacker_policy in {"deepest_manager", "deepest_middle_manager"}:
                inferred_depth = AttackInjectionDepth.MIDDLE_MANAGER
            else:
                inferred_depth = AttackInjectionDepth.LEAF
            self.injection_depth = inferred_depth
        elif inferred_depth is AttackInjectionDepth.MANAGER:
            inferred_depth = AttackInjectionDepth.MIDDLE_MANAGER
            self.injection_depth = inferred_depth

        if self.attack_mode is AttackMode.COMPROMISED_LEAF and inferred_depth is not AttackInjectionDepth.LEAF:
            raise ValueError("attack_mode=compromised_leaf is only compatible with injection_depth=leaf")
        if self.attacker_policy == "deepest_leaf" and inferred_depth is not AttackInjectionDepth.LEAF:
            raise ValueError("attacker_policy=deepest_leaf is only compatible with injection_depth=leaf")
        if self.attacker_policy in {"deepest_manager", "deepest_middle_manager"} and inferred_depth is not AttackInjectionDepth.MIDDLE_MANAGER:
            raise ValueError("manager attacker policies are only compatible with injection_depth=middle_manager")

        if self.attack_mode is None and inferred_depth is AttackInjectionDepth.LEAF:
            self.attack_mode = AttackMode.COMPROMISED_LEAF
        if self.attacker_policy is None:
            self.attacker_policy = "deepest_leaf" if inferred_depth is AttackInjectionDepth.LEAF else "deepest_middle_manager"
        return self


class AttackScenarioConfig(AttackConfig):
    model_config = ConfigDict(extra="forbid")

    name: str
    topology_presets: list[TopologyPreset] = Field(default_factory=list)
    baselines: list[BaselineStrategy] = Field(default_factory=list)


class RunMetadataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    framework_id: str = "coord_harness_v1"
    stage: RunStage = RunStage.CLEAN
    mode: RunMode = RunMode.RESEARCH_STRICT
    description: str | None = None
    output_root: Path = Path("outputs")
    agent_count: int = 5
    total_billed_token_budget: int = 4096
    max_concurrency: int = 25
    capture_model_catalog_snapshot: bool = False
    notes: dict[str, str] = Field(default_factory=dict)
    regime_thresholds: RegimeThresholds = Field(default_factory=RegimeThresholds)
    attack: AttackConfig = Field(default_factory=AttackConfig)

    @field_validator("agent_count")
    @classmethod
    def validate_agent_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent_count must be >= 1")
        return value

    @field_validator("total_billed_token_budget")
    @classmethod
    def validate_budget(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("total_billed_token_budget must be > 0")
        return value

    @field_validator("max_concurrency")
    @classmethod
    def validate_max_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_concurrency must be >= 1")
        return value


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_families: list[BenchmarkFamily]
    topology_presets: list[TopologyPreset]
    message_token_budgets: list[int]
    model_aliases: list[str] = Field(default_factory=list)
    model_tiers: list[PanelTier] = Field(default_factory=list)
    baselines: list[BaselineStrategy]
    seeds: list[int]
    attack_scenarios: list[AttackScenarioConfig] = Field(default_factory=list)

    @field_validator("message_token_budgets")
    @classmethod
    def validate_message_budgets(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("At least one message token budget must be configured.")
        invalid = set(values) - P0A_MESSAGE_TOKEN_PRESETS
        if invalid:
            raise ValueError(f"P0a clean supports only {sorted(P0A_MESSAGE_TOKEN_PRESETS)}; got {sorted(invalid)}")
        return values

    @model_validator(mode="after")
    def validate_model_selection(self) -> "SweepConfig":
        if not self.model_aliases and not self.model_tiers:
            raise ValueError("Sweep must select models via model_aliases or model_tiers.")
        scenario_names = [scenario.name for scenario in self.attack_scenarios]
        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError(f"attack_scenarios must have unique names; got duplicates in {scenario_names}")
        return self


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunMetadataConfig
    benchmarks: list[BenchmarkAdapterConfig]
    model_panel: list[ModelSpec]
    sweep: SweepConfig

    @model_validator(mode="after")
    def validate_references(self) -> "BatchConfig":
        if self.run.stage is RunStage.CLEAN and self.run.attack.enabled:
            raise ValueError("Attack layer must be disabled in clean stage.")
        if self.run.stage is RunStage.STRESS and not self.run.attack.enabled:
            raise ValueError("Stress stage requires attack.enabled=true.")
        if self.run.stage is RunStage.CLEAN and self.sweep.attack_scenarios:
            raise ValueError("attack_scenarios are only valid for stress runs.")

        benchmark_families = {entry.family for entry in self.benchmarks}
        if not set(self.sweep.benchmark_families).issubset(benchmark_families):
            missing = set(self.sweep.benchmark_families) - benchmark_families
            raise ValueError(f"Sweep references undefined benchmark families: {sorted(f.value for f in missing)}")

        aliases = {entry.alias for entry in self.model_panel}
        if not set(self.sweep.model_aliases).issubset(aliases):
            missing_aliases = set(self.sweep.model_aliases) - aliases
            raise ValueError(f"Sweep references undefined model aliases: {sorted(missing_aliases)}")

        available_tiers = {entry.panel_tier for entry in self.model_panel}
        if not set(self.sweep.model_tiers).issubset(available_tiers):
            missing_tiers = set(self.sweep.model_tiers) - available_tiers
            raise ValueError(f"Sweep references undefined model tiers: {sorted(t.value for t in missing_tiers)}")

        available_topologies = set(self.sweep.topology_presets)
        available_baselines = set(self.sweep.baselines)
        for scenario in self.sweep.attack_scenarios:
            if scenario.topology_presets and not set(scenario.topology_presets).issubset(available_topologies):
                missing_topologies = set(scenario.topology_presets) - available_topologies
                raise ValueError(
                    f"Attack scenario {scenario.name} references undefined topology presets: "
                    f"{sorted(item.value for item in missing_topologies)}"
                )
            if scenario.baselines and not set(scenario.baselines).issubset(available_baselines):
                missing_baselines = set(scenario.baselines) - available_baselines
                raise ValueError(
                    f"Attack scenario {scenario.name} references undefined baselines: "
                    f"{sorted(item.value for item in missing_baselines)}"
                )

        for benchmark in self.benchmarks:
            if self.run.stage is RunStage.STRESS:
                if benchmark.stage not in {RunStage.STRESS, RunStage.CLEAN}:
                    raise ValueError(
                        f"Benchmark {benchmark.family.value} stage {benchmark.stage.value} is invalid for stress run"
                    )
            elif benchmark.stage is not self.run.stage:
                raise ValueError(
                    f"Benchmark {benchmark.family.value} stage {benchmark.stage.value} does not match run stage {self.run.stage.value}"
                )

        for model in self.model_panel:
            if self.run.mode is RunMode.RESEARCH_STRICT:
                if model.temperature != 0.0:
                    raise ValueError(f"Clean strict mode requires temperature=0.0; got {model.temperature} for {model.alias}")
                if model.allow_auto_routing:
                    raise ValueError(f"Research strict mode forbids auto routing; got allow_auto_routing=true for {model.alias}")
                if model.provider is ModelProvider.OPENROUTER:
                    blocked_model_ids = {"openrouter/auto", "openrouter/auto-router"}
                    if model.model_id in blocked_model_ids:
                        raise ValueError(f"Research strict mode forbids non-pinned model ids like {model.model_id}")
                    if model.routing.allow_fallbacks:
                        raise ValueError(f"Research strict mode forbids provider fallbacks for {model.alias}")
                    if not model.routing.only and not model.routing.order:
                        raise ValueError(
                            f"Research strict mode requires explicit provider pinning via routing.only or routing.order for {model.alias}"
                        )
            if self.run.mode is RunMode.DEV_CONVENIENCE and model.provider is ModelProvider.OPENROUTER:
                if model.temperature < 0:
                    raise ValueError(f"temperature cannot be negative for {model.alias}")
        return self

    def benchmark_config_for(self, family: BenchmarkFamily) -> BenchmarkAdapterConfig:
        for entry in self.benchmarks:
            if entry.family is family:
                return entry
        raise KeyError(f"Missing benchmark config for {family.value}")

    def model_spec_for(self, alias: str) -> ModelSpec:
        for entry in self.model_panel:
            if entry.alias == alias:
                return entry
        raise KeyError(f"Missing model spec for {alias}")

    def selected_model_aliases(self) -> list[str]:
        aliases = list(self.sweep.model_aliases)
        if self.sweep.model_tiers:
            aliases.extend(
                entry.alias
                for entry in self.model_panel
                if entry.panel_tier in set(self.sweep.model_tiers)
            )
        deduped: list[str] = []
        for alias in aliases:
            if alias not in deduped:
                deduped.append(alias)
        return deduped
