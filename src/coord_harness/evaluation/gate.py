from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coord_harness.core.enums import RegimeLabel


TARGETS = ("help_vs_rest", "collapse_vs_rest", "non_saturation_vs_rest")


@dataclass(frozen=True)
class TrialRow:
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
    F: float | None
    rho: float | None
    B: float | None
    C: float | None
    regime: str


def load_trial_rows(experiment_dir_or_batch_index: str | Path) -> list[TrialRow]:
    path = Path(experiment_dir_or_batch_index)
    batch_index_path = path if path.name == "batch_index.json" else path / "batch_index.json"
    payload = json.loads(batch_index_path.read_text(encoding="utf-8"))
    rows: list[TrialRow] = []
    for entry in payload["trials"]:
        if entry["baseline"] == "sa_star" or entry["regime"] is None:
            continue
        summary = json.loads(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        rows.append(
            TrialRow(
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
                F=summary["derived"]["F"],
                rho=summary["derived"]["rho"],
                B=summary["derived"]["B"],
                C=summary["derived"]["C"],
                regime=entry["regime"],
            )
        )
    return rows


def _binary_label(row: TrialRow, target: str) -> int:
    if target == "help_vs_rest":
        return 1 if row.regime == RegimeLabel.HELP.value else 0
    if target == "collapse_vs_rest":
        return 1 if row.regime == RegimeLabel.COLLAPSE.value else 0
    if target == "non_saturation_vs_rest":
        return 0 if row.regime == RegimeLabel.SATURATION.value else 1
    raise ValueError(f"Unknown target: {target}")


class FeatureEncoder:
    def __init__(self, *, include_core_metrics: bool):
        self.include_core_metrics = include_core_metrics
        self.numeric_names = [
            "budget",
            "agent_count",
            "total_budget",
            "mean_message_tokens",
            "total_messages",
        ]
        if include_core_metrics:
            self.numeric_names.extend(["F", "rho", "B", "C"])
        self.categorical_names = ["topology", "model_alias", "baseline"]
        self.category_values: dict[str, list[str]] = {}
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None

    def fit(self, rows: list[TrialRow]) -> None:
        self.category_values = {
            name: sorted({getattr(row, name) for row in rows})
            for name in self.categorical_names
        }
        numeric = np.array([self._numeric_values(row) for row in rows], dtype=float)
        self.numeric_mean = numeric.mean(axis=0)
        self.numeric_std = numeric.std(axis=0)
        self.numeric_std[self.numeric_std == 0] = 1.0

    def transform(self, rows: list[TrialRow]) -> np.ndarray:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("Encoder must be fit before transform.")
        numeric = np.array([self._numeric_values(row) for row in rows], dtype=float)
        numeric = (numeric - self.numeric_mean) / self.numeric_std
        categorical_parts = []
        for name in self.categorical_names:
            categories = self.category_values.get(name, [])
            matrix = np.zeros((len(rows), len(categories)), dtype=float)
            for row_index, row in enumerate(rows):
                try:
                    col = categories.index(getattr(row, name))
                except ValueError:
                    continue
                matrix[row_index, col] = 1.0
            categorical_parts.append(matrix)
        features = [numeric] + categorical_parts
        return np.concatenate(features, axis=1) if features else np.zeros((len(rows), 0), dtype=float)

    def _numeric_values(self, row: TrialRow) -> list[float]:
        values = [
            float(row.budget),
            float(row.agent_count),
            float(row.total_budget),
            float(row.mean_message_tokens),
            float(row.total_messages),
        ]
        if self.include_core_metrics:
            values.extend(
                [
                    float(row.F or 0.0),
                    float(row.rho or 0.0),
                    float(row.B or 0.0),
                    float(row.C or 0.0),
                ]
            )
        return values


class LogisticRegressor:
    def __init__(self, learning_rate: float = 0.1, l2: float = 0.01, steps: int = 400):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.steps = steps
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0
        for _ in range(self.steps):
            logits = X @ self.weights + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            error = probs - y
            grad_w = (X.T @ error) / len(X) + self.l2 * self.weights
            grad_b = float(error.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be fit before predict.")
        logits = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
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


def evaluate_gate(
    experiment_dir_or_batch_index: str | Path,
    *,
    primary_target: str = "help_vs_rest",
    min_auc: float = 0.75,
    min_delta: float = 0.10,
) -> dict[str, Any]:
    if primary_target not in TARGETS:
        raise ValueError(f"primary_target must be one of {TARGETS}")
    rows = load_trial_rows(experiment_dir_or_batch_index)
    families = sorted({row.family for row in rows})

    holdout_results: dict[str, dict[str, dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        holdout_results[target] = {}
        heuristic_aucs: list[float] = []
        core_aucs: list[float] = []
        for family in families:
            train_rows = [row for row in rows if row.family != family]
            test_rows = [row for row in rows if row.family == family]
            y_train = np.array([_binary_label(row, target) for row in train_rows], dtype=float)
            y_test = np.array([_binary_label(row, target) for row in test_rows], dtype=float)
            if len(train_rows) == 0 or len(test_rows) == 0 or len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
                holdout_results[target][family] = {"status": "insufficient_label_variation"}
                continue

            heuristic_encoder = FeatureEncoder(include_core_metrics=False)
            heuristic_encoder.fit(train_rows)
            X_train_heur = heuristic_encoder.transform(train_rows)
            X_test_heur = heuristic_encoder.transform(test_rows)
            heuristic_model = LogisticRegressor()
            heuristic_model.fit(X_train_heur, y_train)
            heuristic_auc = compute_auroc(y_test, heuristic_model.predict_proba(X_test_heur))

            core_encoder = FeatureEncoder(include_core_metrics=True)
            core_encoder.fit(train_rows)
            X_train_core = core_encoder.transform(train_rows)
            X_test_core = core_encoder.transform(test_rows)
            core_model = LogisticRegressor()
            core_model.fit(X_train_core, y_train)
            core_auc = compute_auroc(y_test, core_model.predict_proba(X_test_core))

            holdout_results[target][family] = {
                "status": "ok",
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "heuristic_auroc": heuristic_auc,
                "core_auroc": core_auc,
                "delta_auroc": round((core_auc or 0.0) - (heuristic_auc or 0.0), 6)
                if heuristic_auc is not None and core_auc is not None
                else None,
            }
            if heuristic_auc is not None:
                heuristic_aucs.append(heuristic_auc)
            if core_auc is not None:
                core_aucs.append(core_auc)

        mean_heuristic = round(float(np.mean(heuristic_aucs)), 6) if heuristic_aucs else None
        mean_core = round(float(np.mean(core_aucs)), 6) if core_aucs else None
        aggregate[target] = {
            "mean_heuristic_auroc": mean_heuristic,
            "mean_core_auroc": mean_core,
            "delta_auroc": round(mean_core - mean_heuristic, 6)
            if mean_heuristic is not None and mean_core is not None
            else None,
        }

    primary = aggregate[primary_target]
    gate_pass = bool(
        primary["mean_core_auroc"] is not None
        and primary["delta_auroc"] is not None
        and primary["mean_core_auroc"] >= min_auc
        and primary["delta_auroc"] >= min_delta
    )
    return {
        "primary_target": primary_target,
        "thresholds": {"min_auc": min_auc, "min_delta": min_delta},
        "sample_count": len(rows),
        "families": families,
        "targets": aggregate,
        "holdouts": holdout_results,
        "gate_pass": gate_pass,
    }


def write_gate_report(
    experiment_dir_or_batch_index: str | Path,
    *,
    primary_target: str = "help_vs_rest",
    min_auc: float = 0.75,
    min_delta: float = 0.10,
) -> Path:
    path = Path(experiment_dir_or_batch_index)
    experiment_dir = path.parent if path.name == "batch_index.json" else path
    report = evaluate_gate(
        experiment_dir_or_batch_index,
        primary_target=primary_target,
        min_auc=min_auc,
        min_delta=min_delta,
    )
    output_path = experiment_dir / "gate_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
