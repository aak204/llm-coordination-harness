from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from coord_harness.analysis.attack_analysis import write_attack_analysis


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _annotate_vertical_bars(ax, bars, *, fmt: str = "{:.2f}", dy: float = 0.015) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _annotate_horizontal_bars(ax, bars, *, fmt: str = "{:.3f}", dx: float = 0.01) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + dx,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            ha="left",
            va="center",
            fontsize=8,
        )


def _short_model_name(alias: str) -> str:
    return {
        "qwen35_plus_0215": "Qwen",
        "gemini31_flash_lite_preview": "Gemini",
    }.get(alias, alias)


def _short_family_name(name: str) -> str:
    return {"agentsnet_mini": "AgentsNet", "craft_mini": "CRAFT"}.get(name, name)


def _plot_feature_importance(experiment_dir: Path, output_dir: Path) -> None:
    report = _load_json(experiment_dir / "predictor_analysis.json")
    target = report["targets"]["help_vs_rest"]
    heuristic = target["global_feature_importance"]["heuristic_rf"]["grouped"][:10]
    core = target["global_feature_importance"]["core_rf"]["grouped"][:10]

    feature_names = []
    importance_map: dict[str, dict[str, float]] = {}
    for item in heuristic:
        feature_names.append(item["feature_group"])
        importance_map.setdefault(item["feature_group"], {})["heuristic"] = item["importance"]
    for item in core:
        if item["feature_group"] not in feature_names:
            feature_names.append(item["feature_group"])
        importance_map.setdefault(item["feature_group"], {})["core"] = item["importance"]

    heuristic_values = [importance_map[name].get("heuristic", 0.0) for name in feature_names]
    core_values = [importance_map[name].get("core", 0.0) for name in feature_names]
    y = list(range(len(feature_names)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars_h = ax.barh([idx - 0.18 for idx in y], heuristic_values, height=0.34, label="Heuristic RF", color="#8C7A6B")
    bars_c = ax.barh([idx + 0.18 for idx in y], core_values, height=0.34, label="Core RF", color="#246A73")
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel("Grouped Feature Importance")
    ax.set_title("Feature Importance: Heuristic vs Core (`help_vs_rest`)")
    ax.legend(loc="lower right")
    fig.text(
        0.01,
        0.005,
        "Global grouped importances; RandomForest; train excludes budget == 0; release run p0a-calibrated-full-live.",
        fontsize=9,
    )
    _annotate_horizontal_bars(ax, bars_h)
    _annotate_horizontal_bars(ax, bars_c)
    _save_figure(fig, output_dir, "feature_importance_help_vs_rest")


def _topology_rows(experiment_dir: Path, *, baseline: str = "ma_ft", budget: int = 96) -> list[dict]:
    batch = _load_json(experiment_dir / "batch_index.json")
    rows = []
    for entry in batch["trials"]:
        if entry["baseline"] != baseline or entry["message_token_budget"] != budget:
            continue
        summary = _load_json(Path(entry["summary_path"]))
        rows.append(
            {
                "label": f"{_short_family_name(entry['benchmark_family'])} / {_short_model_name(entry['model_alias'])}",
                "topology": "Balanced Tree" if entry["topology_preset"] == "balanced_tree" else "Star",
                "score": entry["score_mean"],
                "F": summary["derived"]["F"] or 0.0,
                "B": summary["derived"]["B"] or 0.0,
            }
        )
    return rows


def _plot_topology_penalty(experiment_dir: Path, output_dir: Path) -> None:
    rows = _topology_rows(experiment_dir, baseline="ma_ft", budget=96)
    labels = sorted({row["label"] for row in rows})
    metrics = ["score", "F", "B"]
    colors = {"Star": "#2A9D8F", "Balanced Tree": "#E76F51"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = list(range(len(labels)))
    for ax, metric in zip(axes, metrics):
        star_values = [next(row[metric] for row in rows if row["label"] == label and row["topology"] == "Star") for label in labels]
        tree_values = [next(row[metric] for row in rows if row["label"] == label and row["topology"] == "Balanced Tree") for label in labels]
        bars_s = ax.bar([idx - 0.18 for idx in x], star_values, width=0.35, color=colors["Star"], label="Star")
        bars_t = ax.bar([idx + 0.18 for idx in x], tree_values, width=0.35, color=colors["Balanced Tree"], label="Balanced Tree")
        _annotate_vertical_bars(ax, bars_s)
        _annotate_vertical_bars(ax, bars_t)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.08)
        if metric == "score":
            ax.legend(loc="lower left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.suptitle("Topology Penalty: Star vs Balanced Tree (`MA-FT`, budget 96)")
    fig.text(0.01, 0.005, "MA-FT only; budget 96 only; n = 8 cells (2 families x 2 models x 2 seeds).", fontsize=9)
    _save_figure(fig, output_dir, "topology_penalty_budget96_maft")


def _plot_topology_delta(experiment_dir: Path, output_dir: Path) -> None:
    rows = _topology_rows(experiment_dir, baseline="ma_ft", budget=96)
    labels = sorted({row["label"] for row in rows})
    metrics = ["score", "F", "B"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = list(range(len(labels)))
    for ax, metric in zip(axes, metrics):
        deltas = []
        for label in labels:
            star = next(row[metric] for row in rows if row["label"] == label and row["topology"] == "Star")
            tree = next(row[metric] for row in rows if row["label"] == label and row["topology"] == "Balanced Tree")
            deltas.append(tree - star)
        bars = ax.bar(x, deltas, width=0.5, color=["#E76F51" if value < 0 else "#2A9D8F" for value in deltas])
        _annotate_vertical_bars(ax, bars, fmt="{:+.2f}", dy=0.01 if max(deltas, default=0) >= 0 else -0.03)
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_ylabel(f"Δ {metric}")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.suptitle("Topology Delta: Balanced Tree - Star (`MA-FT`, budget 96)")
    fig.text(0.01, 0.005, "Negative values indicate a tree penalty; MA-FT only; budget 96 only; n = 8 cells.", fontsize=9)
    _save_figure(fig, output_dir, "topology_delta_budget96_maft")


def _plot_predictor_holdout(experiment_dir: Path, output_dir: Path) -> None:
    report = _load_json(experiment_dir / "predictor_analysis.json")
    targets = ["help_vs_rest", "non_saturation_vs_rest"]
    families = report["families"]
    bars = []
    deltas = []
    for target in targets:
        for family in families:
            holdout = report["targets"][target]["holdouts"][family]
            heur = holdout["heuristic_rf"]["auroc_nonzero"]
            core = holdout["core_rf"]["auroc_nonzero"]
            bars.append(
                {
                    "label": f"{target}\n{_short_family_name(family)}",
                    "heuristic": heur,
                    "core": core,
                }
            )
            deltas.append(core - heur if heur is not None and core is not None else None)

    x = list(range(len(bars)))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_h = ax.bar([idx - 0.18 for idx in x], [bar["heuristic"] for bar in bars], width=0.35, color="#8C7A6B", label="Heuristic RF")
    bars_c = ax.bar([idx + 0.18 for idx in x], [bar["core"] for bar in bars], width=0.35, color="#246A73", label="Core RF")
    _annotate_vertical_bars(ax, bars_h)
    _annotate_vertical_bars(ax, bars_c)
    for idx, delta in enumerate(deltas):
        if delta is None:
            continue
        top = max(bars[idx]["heuristic"], bars[idx]["core"])
        ax.text(idx, top + 0.04, f"{delta:+.03f}", ha="center", va="bottom", fontsize=9, color="#222222")
    ax.axhline(0.75, color="#444444", linestyle="--", linewidth=1.2, label="Gate AUROC 0.75")
    ax.set_xticks(x)
    ax.set_xticklabels([bar["label"] for bar in bars])
    ax.set_ylabel("Held-out AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-out Family AUROC: Heuristic vs Core")
    ax.legend(loc="lower right")
    fig.text(0.01, 0.005, "Held-out family; nonzero-budget test only; train excludes budget == 0; deltas shown above each pair.", fontsize=9)
    _save_figure(fig, output_dir, "predictor_holdout_auroc_nonzero")


def _stress_rows(stress_dir: Path, clean_dir: Path) -> list[dict]:
    stress = _load_json(stress_dir / "batch_index.json")
    clean = _load_json(clean_dir / "batch_index.json")
    clean_map = {}
    for entry in clean["trials"]:
        if entry["seed"] != 7:
            continue
        key = (
            entry["benchmark_family"],
            entry["topology_preset"],
            entry["message_token_budget"],
            entry["model_alias"],
            entry["baseline"],
        )
        clean_map[key] = entry

    rows = []
    for entry in stress["trials"]:
        summary = _load_json(Path(entry["summary_path"]))
        key = (
            entry["benchmark_family"],
            entry["topology_preset"],
            entry["message_token_budget"],
            entry["model_alias"],
            entry["baseline"],
        )
        clean_score = clean_map[key]["score_mean"]
        rows.append(
            {
                "label": f"{_short_family_name(entry['benchmark_family'])} / {_short_model_name(entry['model_alias'])} / {entry['baseline']}",
                "topology": "Balanced Tree" if entry["topology_preset"] == "balanced_tree" else "Star",
                "delta_score": entry["score_mean"] - clean_score,
                "infection": summary["outcomes"]["infection_spread_rate"] or 0.0,
                "attack_success": summary["outcomes"]["attack_success_rate"] or 0.0,
                "quarantine_strength": summary["outcomes"].get("quarantine_strength") or 0.0,
                "f_delta": (summary.get("attack_analysis") or {}).get("F_delta_vs_clean"),
            }
        )
    return rows


def _load_attack_analysis_report(stress_dir: Path, clean_dir: Path) -> dict:
    report_path = stress_dir / "attack_analysis_report.json"
    if not report_path.exists():
        report_path = write_attack_analysis(stress_dir, clean_dir)
    return _load_json(report_path)


def _plot_attack_score_delta(stress_dir: Path, clean_dir: Path, output_dir: Path) -> None:
    rows = _stress_rows(stress_dir, clean_dir)
    labels = sorted({row["label"] for row in rows})
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6))
    star_values = [next(row["delta_score"] for row in rows if row["label"] == label and row["topology"] == "Star") for label in labels]
    tree_values = [next(row["delta_score"] for row in rows if row["label"] == label and row["topology"] == "Balanced Tree") for label in labels]
    bars_s = ax.bar([idx - 0.18 for idx in x], star_values, width=0.35, color="#2A9D8F", label="Star")
    bars_t = ax.bar([idx + 0.18 for idx in x], tree_values, width=0.35, color="#E76F51", label="Balanced Tree")
    _annotate_vertical_bars(ax, bars_s, fmt="{:+.2f}", dy=0.01)
    _annotate_vertical_bars(ax, bars_t, fmt="{:+.2f}", dy=0.01)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Stress - Clean Score")
    ax.set_title("P0b Attack Score Delta vs Clean")
    ax.legend(loc="lower left")
    fig.text(0.01, 0.005, "Compared to clean seed-7 reference; Star and Balanced Tree shown separately.", fontsize=9)
    _save_figure(fig, output_dir, "attack_score_delta_vs_clean")


def _plot_attack_infection(stress_dir: Path, clean_dir: Path, output_dir: Path) -> None:
    rows = _stress_rows(stress_dir, clean_dir)
    labels = sorted({row["label"] for row in rows})
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6))
    star_values = [next(row["infection"] for row in rows if row["label"] == label and row["topology"] == "Star") for label in labels]
    tree_values = [next(row["infection"] for row in rows if row["label"] == label and row["topology"] == "Balanced Tree") for label in labels]
    bars_s = ax.bar([idx - 0.18 for idx in x], star_values, width=0.35, color="#2A9D8F", label="Star")
    bars_t = ax.bar([idx + 0.18 for idx in x], tree_values, width=0.35, color="#E76F51", label="Balanced Tree")
    _annotate_vertical_bars(ax, bars_s)
    _annotate_vertical_bars(ax, bars_t)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Infection Spread Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("P0b Infection Spread")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.005, "Stress-only metric; higher means more non-attacker nodes adopted the malicious answer.", fontsize=9)
    _save_figure(fig, output_dir, "attack_infection_spread")


def _plot_attack_success(stress_dir: Path, clean_dir: Path, output_dir: Path) -> None:
    rows = _stress_rows(stress_dir, clean_dir)
    labels = sorted({row["label"] for row in rows})
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6))
    star_values = [next(row["attack_success"] for row in rows if row["label"] == label and row["topology"] == "Star") for label in labels]
    tree_values = [next(row["attack_success"] for row in rows if row["label"] == label and row["topology"] == "Balanced Tree") for label in labels]
    bars_s = ax.bar([idx - 0.18 for idx in x], star_values, width=0.35, color="#2A9D8F", label="Star")
    bars_t = ax.bar([idx + 0.18 for idx in x], tree_values, width=0.35, color="#E76F51", label="Balanced Tree")
    _annotate_vertical_bars(ax, bars_s)
    _annotate_vertical_bars(ax, bars_t)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("P0b Attack Success")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.005, "Stress-only metric; higher means the malicious answer captured the final system output more often.", fontsize=9)
    _save_figure(fig, output_dir, "attack_success_rate")


def _plot_attack_f_tradeoff(stress_dir: Path, clean_dir: Path, output_dir: Path) -> None:
    report = _load_attack_analysis_report(stress_dir, clean_dir)
    rows = [
        row
        for row in report["rows"]
        if row["F_delta_vs_clean"] is not None and row["infection_spread_rate"] is not None
    ]
    if not rows:
        return

    colors = {"star": "#2A9D8F", "balanced_tree": "#E76F51"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for row in rows:
        x_value = row["F_delta_vs_clean"]
        y_value = row["infection_spread_rate"]
        ax.scatter(
            x_value,
            y_value,
            s=90,
            color=colors.get(row["topology_preset"], "#444444"),
            alpha=0.9,
        )
        point_label = (
            f"{_short_family_name(row['benchmark_family'])}/"
            f"{_short_model_name(row['model_alias'])}/"
            f"{row['message_token_budget']}"
        )
        ax.text(x_value + 0.006, y_value + 0.01, point_label, fontsize=8)

    correlation = report["correlations"]["overall"]
    xs = [row["F_delta_vs_clean"] for row in rows]
    ys = [row["infection_spread_rate"] for row in rows]
    slope = correlation.get("linear_slope")
    if slope is not None and len(xs) >= 2:
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        intercept = mean_y - slope * mean_x
        start_x = min(xs)
        end_x = max(xs)
        ax.plot(
            [start_x, end_x],
            [slope * start_x + intercept, slope * end_x + intercept],
            color="#264653",
            linewidth=1.8,
        )

    ax.set_xlabel("F Drop vs Clean (clean F - stress F)")
    ax.set_ylabel("Infection Spread Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Natural Quarantine Tradeoff: Useful Fact Loss vs Viral Spread")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=colors["star"], label="Star", markersize=8),
        Line2D([], [], marker="o", linestyle="", color=colors["balanced_tree"], label="Balanced Tree", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.text(
        0.01,
        0.005,
        (
            "Each point is one stress cell with a matched clean control; "
            f"overall Pearson r = {correlation.get('pearson_r')}, slope = {correlation.get('linear_slope')}."
        ),
        fontsize=9,
    )
    _save_figure(fig, output_dir, "attack_f_delta_vs_infection_scatter")


def generate_plots(
    clean_experiment_dir: str | Path = "outputs/p0a-calibrated-full-live",
    output_dir: str | Path = "docs/figures",
    stress_experiment_dir: str | Path = "outputs/p0b-systematic-attacks",
) -> Path:
    clean_path = Path(clean_experiment_dir)
    stress_path = Path(stress_experiment_dir)
    output_path = Path(output_dir)
    _ensure_dir(output_path)
    _plot_feature_importance(clean_path, output_path)
    _plot_topology_penalty(clean_path, output_path)
    _plot_topology_delta(clean_path, output_path)
    _plot_predictor_holdout(clean_path, output_path)
    if stress_path.exists():
        _plot_attack_score_delta(stress_path, clean_path, output_path)
        _plot_attack_infection(stress_path, clean_path, output_path)
        _plot_attack_success(stress_path, clean_path, output_path)
        _plot_attack_f_tradeoff(stress_path, clean_path, output_path)
    return output_path


if __name__ == "__main__":
    print(generate_plots())
