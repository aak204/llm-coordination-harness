from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from coord_harness.core.enums import RegimeLabel


TARGETS = ("help_vs_rest", "collapse_vs_rest", "non_saturation_vs_rest")


@dataclass(frozen=True)
class PredictorRow:
    trial_id: str
    family: str
    topology: str
    model_alias: str
    baseline: str
    budget: int
    agent_count: int
    total_budget: int
    mean_message_tokens: float
    total_messages: int
    mean_billed_tokens: float
    mean_inter_agent_tokens: float
    F: float | None
    rho: float | None
    B: float | None
    C: float | None
    regime: str


def _binary_label(row: PredictorRow, target: str) -> int:
    if target == "help_vs_rest":
        return 1 if row.regime == RegimeLabel.HELP.value else 0
    if target == "collapse_vs_rest":
        return 1 if row.regime == RegimeLabel.COLLAPSE.value else 0
    if target == "non_saturation_vs_rest":
        return 0 if row.regime == RegimeLabel.SATURATION.value else 1
    raise ValueError(f"Unknown target: {target}")


def _compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - positives * (positives + 1) / 2) / (positives * negatives)
    return round(float(auc), 6)


def _group_feature_name(name: str) -> str:
    if "=" in name:
        return name.split("=", 1)[0]
    if name.endswith("_value"):
        return name[:-6]
    if name.endswith("_missing"):
        return name[:-8]
    return name


def _build_feature_dict(row: PredictorRow, *, include_core: bool) -> dict[str, float]:
    features: dict[str, float] = {
        "budget": float(row.budget),
        "agent_count": float(row.agent_count),
        "total_budget": float(row.total_budget),
        "mean_message_tokens": float(row.mean_message_tokens),
        "total_messages": float(row.total_messages),
        "mean_billed_tokens": float(row.mean_billed_tokens),
        "mean_inter_agent_tokens": float(row.mean_inter_agent_tokens),
        f"topology={row.topology}": 1.0,
        f"model_alias={row.model_alias}": 1.0,
        f"baseline={row.baseline}": 1.0,
    }
    if include_core:
        for name, value in [("F", row.F), ("rho", row.rho), ("B", row.B), ("C", row.C)]:
            features[f"{name}_value"] = 0.0 if value is None else float(value)
            features[f"{name}_missing"] = 1.0 if value is None else 0.0
    return features


def load_predictor_rows(experiment_dir_or_batch_index: str | Path) -> list[PredictorRow]:
    path = Path(experiment_dir_or_batch_index)
    batch_index_path = path if path.name == "batch_index.json" else path / "batch_index.json"
    payload = json.loads(batch_index_path.read_text(encoding="utf-8"))
    rows: list[PredictorRow] = []
    for entry in payload["trials"]:
        if entry["baseline"] == "sa_star" or entry["regime"] is None:
            continue
        summary = json.loads(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        rows.append(
            PredictorRow(
                trial_id=entry["trial_id"],
                family=entry["benchmark_family"],
                topology=entry["topology_preset"],
                model_alias=entry["model_alias"],
                baseline=entry["baseline"],
                budget=entry["message_token_budget"],
                agent_count=summary["topology"]["agent_count"],
                total_budget=summary["budget"]["total_billed_token_budget_per_task"],
                mean_message_tokens=summary["derived"]["mean_message_tokens"],
                total_messages=summary["derived"]["total_messages"],
                mean_billed_tokens=summary["budget"]["mean_billed_tokens"],
                mean_inter_agent_tokens=summary["budget"]["mean_inter_agent_tokens"],
                F=summary["derived"]["F"],
                rho=summary["derived"]["rho"],
                B=summary["derived"]["B"],
                C=summary["derived"]["C"],
                regime=entry["regime"],
            )
        )
    return rows


def _fit_random_forest(rows: list[PredictorRow], target: str, *, include_core: bool, random_state: int) -> tuple[RandomForestClassifier, DictVectorizer]:
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform([_build_feature_dict(row, include_core=include_core) for row in rows])
    y = np.array([_binary_label(row, target) for row in rows], dtype=int)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)
    return model, vectorizer


def _feature_importance_report(model: RandomForestClassifier, vectorizer: DictVectorizer) -> dict[str, Any]:
    names = vectorizer.feature_names_
    importances = model.feature_importances_
    raw = sorted(
        [{"feature": name, "importance": round(float(score), 6)} for name, score in zip(names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    grouped: dict[str, float] = {}
    for name, score in zip(names, importances):
        grouped_name = _group_feature_name(name)
        grouped[grouped_name] = grouped.get(grouped_name, 0.0) + float(score)
    grouped_sorted = sorted(
        [{"feature_group": name, "importance": round(score, 6)} for name, score in grouped.items()],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "grouped": grouped_sorted,
        "raw_top_15": raw[:15],
    }


def analyze_predictor(experiment_dir_or_batch_index: str | Path) -> dict[str, Any]:
    rows = load_predictor_rows(experiment_dir_or_batch_index)
    families = sorted({row.family for row in rows})

    results: dict[str, Any] = {
        "sample_count": len(rows),
        "families": families,
        "filters": {
            "train_excludes_budget_zero": True,
            "reports_include_full_test_and_nonzero_test": True,
        },
        "targets": {},
    }

    for target_index, target in enumerate(TARGETS, start=1):
        target_result: dict[str, Any] = {
            "holdouts": {},
            "global_feature_importance": {},
        }
        for include_core, label in [(False, "heuristic_rf"), (True, "core_rf")]:
            train_rows_global = [row for row in rows if row.budget != 0]
            if len({ _binary_label(row, target) for row in train_rows_global }) >= 2:
                model, vectorizer = _fit_random_forest(
                    train_rows_global,
                    target,
                    include_core=include_core,
                    random_state=100 + target_index + (10 if include_core else 0),
                )
                target_result["global_feature_importance"][label] = _feature_importance_report(model, vectorizer)
            else:
                target_result["global_feature_importance"][label] = {"status": "insufficient_label_variation"}

        for family in families:
            holdout: dict[str, Any] = {}
            train_rows = [row for row in rows if row.family != family and row.budget != 0]
            test_rows_all = [row for row in rows if row.family == family]
            test_rows_nonzero = [row for row in test_rows_all if row.budget != 0]
            holdout["train_count"] = len(train_rows)
            holdout["test_count_all"] = len(test_rows_all)
            holdout["test_count_nonzero"] = len(test_rows_nonzero)

            for include_core, label in [(False, "heuristic_rf"), (True, "core_rf")]:
                if len(train_rows) == 0 or len({ _binary_label(row, target) for row in train_rows }) < 2:
                    holdout[label] = {"status": "insufficient_train_label_variation"}
                    continue
                model, vectorizer = _fit_random_forest(
                    train_rows,
                    target,
                    include_core=include_core,
                    random_state=200 + target_index + (10 if include_core else 0),
                )

                metrics: dict[str, Any] = {"status": "ok"}
                for test_label, test_rows in [("all", test_rows_all), ("nonzero", test_rows_nonzero)]:
                    if len(test_rows) == 0:
                        metrics[f"auroc_{test_label}"] = None
                        continue
                    y_test = np.array([_binary_label(row, target) for row in test_rows], dtype=int)
                    if len(set(y_test.tolist())) < 2:
                        metrics[f"auroc_{test_label}"] = None
                        metrics[f"status_{test_label}"] = "insufficient_test_label_variation"
                        continue
                    X_test = vectorizer.transform([_build_feature_dict(row, include_core=include_core) for row in test_rows])
                    probs = model.predict_proba(X_test)[:, 1]
                    metrics[f"auroc_{test_label}"] = _compute_auroc(y_test, probs)
                holdout[label] = metrics

            heur = holdout.get("heuristic_rf", {})
            core = holdout.get("core_rf", {})
            if heur.get("auroc_nonzero") is not None and core.get("auroc_nonzero") is not None:
                holdout["delta_core_minus_heuristic_nonzero"] = round(core["auroc_nonzero"] - heur["auroc_nonzero"], 6)
            else:
                holdout["delta_core_minus_heuristic_nonzero"] = None
            if heur.get("auroc_all") is not None and core.get("auroc_all") is not None:
                holdout["delta_core_minus_heuristic_all"] = round(core["auroc_all"] - heur["auroc_all"], 6)
            else:
                holdout["delta_core_minus_heuristic_all"] = None
            target_result["holdouts"][family] = holdout

        results["targets"][target] = target_result

    return results


def write_predictor_analysis(experiment_dir_or_batch_index: str | Path) -> Path:
    path = Path(experiment_dir_or_batch_index)
    experiment_dir = path.parent if path.name == "batch_index.json" else path
    report = analyze_predictor(experiment_dir_or_batch_index)
    output_path = experiment_dir / "predictor_analysis.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
