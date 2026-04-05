from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coord_harness.logging.schema import AttackAnalysisSection, TrialSummary


def _comparison_key(summary: TrialSummary) -> tuple[str, str, int, str, bool, str, int]:
    return (
        summary.benchmark.family,
        summary.topology.preset,
        summary.budget.message_token_budget,
        summary.model.alias,
        summary.run.enable_reasoning,
        summary.run.baseline,
        summary.run.seed,
    )


def _load_summaries(experiment_dir_or_batch_index: str | Path) -> list[tuple[TrialSummary, Path]]:
    path = Path(experiment_dir_or_batch_index)
    batch_index_path = path if path.name == "batch_index.json" else path / "batch_index.json"
    batch_index = json.loads(batch_index_path.read_text(encoding="utf-8"))
    results: list[tuple[TrialSummary, Path]] = []
    for entry in batch_index["trials"]:
        summary_path = Path(entry["summary_path"])
        summary = TrialSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
        results.append((summary, summary_path))
    return results


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return round(cov / math.sqrt(var_x * var_y), 6)


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    if var_x == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return round(cov / var_x, 6)


def _correlation_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if row["F_delta_vs_clean"] is not None and row["infection_spread_rate"] is not None
    ]
    xs = [float(row["F_delta_vs_clean"]) for row in valid_rows]
    ys = [float(row["infection_spread_rate"]) for row in valid_rows]
    return {
        "point_count": len(valid_rows),
        "pearson_r": _pearson(xs, ys),
        "linear_slope": _linear_slope(xs, ys),
    }


def write_attack_analysis(
    stress_experiment_dir_or_batch_index: str | Path,
    clean_experiment_dir_or_batch_index: str | Path,
) -> Path:
    stress_path = Path(stress_experiment_dir_or_batch_index)
    stress_experiment_dir = stress_path.parent if stress_path.name == "batch_index.json" else stress_path
    clean_summaries = _load_summaries(clean_experiment_dir_or_batch_index)
    clean_map = {_comparison_key(summary): (summary, summary_path) for summary, summary_path in clean_summaries}
    rows: list[dict[str, Any]] = []

    for summary, summary_path in _load_summaries(stress_experiment_dir_or_batch_index):
        key = _comparison_key(summary)
        clean_summary_entry = clean_map.get(key)
        notes: list[str] = []
        clean_summary = None
        clean_summary_path = None
        clean_f = None
        f_delta = None
        score_delta = None
        if clean_summary_entry is None:
            notes.append("No matching clean reference trial found.")
        else:
            clean_summary, clean_summary_path = clean_summary_entry
            clean_f = clean_summary.derived.F
            if clean_summary.derived.F is not None and summary.derived.F is not None:
                f_delta = round(clean_summary.derived.F - summary.derived.F, 6)
            else:
                notes.append("F delta unavailable because clean or stress F is undefined.")
            score_delta = round(summary.outcomes.score_mean - clean_summary.outcomes.score_mean, 6)

        quarantine_strength = summary.outcomes.quarantine_strength
        if quarantine_strength is None and summary.outcomes.infection_spread_rate is not None:
            quarantine_strength = round(1.0 - float(summary.outcomes.infection_spread_rate), 6)
            summary.outcomes.quarantine_strength = quarantine_strength

        summary.attack_analysis = AttackAnalysisSection(
            clean_reference_experiment_id=clean_summary.run.experiment_id if clean_summary else None,
            clean_reference_experiment_dir=str(Path(clean_experiment_dir_or_batch_index).resolve()),
            clean_reference_summary_path=str(clean_summary_path.resolve()) if clean_summary_path else None,
            clean_F=clean_f,
            stress_F=summary.derived.F,
            F_delta_vs_clean=f_delta,
            score_delta_vs_clean=score_delta,
            infection_spread_rate=summary.outcomes.infection_spread_rate,
            attack_success_rate=summary.outcomes.attack_success_rate,
            quarantine_strength=quarantine_strength,
            enable_reasoning=summary.run.enable_reasoning,
            attack_scenario=summary.run.attack_scenario,
            attack_injection_depth=summary.run.attack_injection_depth,
            notes=notes,
        )
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        rows.append(
            {
                "trial_id": summary.run.trial_id,
                "benchmark_family": summary.benchmark.family,
                "topology_preset": summary.topology.preset,
                "message_token_budget": summary.budget.message_token_budget,
                "model_alias": summary.model.alias,
                "enable_reasoning": summary.run.enable_reasoning,
                "baseline": summary.run.baseline,
                "seed": summary.run.seed,
                "clean_F": clean_f,
                "stress_F": summary.derived.F,
                "F_delta_vs_clean": f_delta,
                "infection_spread_rate": summary.outcomes.infection_spread_rate,
                "attack_success_rate": summary.outcomes.attack_success_rate,
                "quarantine_strength": quarantine_strength,
                "score_delta_vs_clean": score_delta,
                "attack_scenario": summary.run.attack_scenario,
                "attack_injection_depth": summary.run.attack_injection_depth,
            }
        )

    by_topology: dict[str, list[dict[str, Any]]] = {}
    by_budget: dict[str, list[dict[str, Any]]] = {}
    by_attack_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_topology.setdefault(row["topology_preset"], []).append(row)
        by_budget.setdefault(str(row["message_token_budget"]), []).append(row)
        by_attack_scenario.setdefault(row["attack_scenario"] or "default", []).append(row)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "stress_experiment_dir": str(stress_experiment_dir.resolve()),
        "clean_experiment_dir": str(Path(clean_experiment_dir_or_batch_index).resolve()),
        "row_count": len(rows),
        "correlations": {
            "overall": _correlation_payload(rows),
            "by_topology": {name: _correlation_payload(group) for name, group in sorted(by_topology.items())},
            "by_budget": {name: _correlation_payload(group) for name, group in sorted(by_budget.items())},
            "by_attack_scenario": {
                name: _correlation_payload(group) for name, group in sorted(by_attack_scenario.items())
            },
        },
        "rows": rows,
    }
    output_path = stress_experiment_dir / "attack_analysis_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
