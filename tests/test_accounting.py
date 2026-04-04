from __future__ import annotations

from pathlib import Path

import yaml

from coord_harness.runner.executor import run_batch


def test_zero_message_budget_produces_zero_inter_agent_tokens_and_zero_messages(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/p0a_clean_smoke.yaml").read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path)
    payload["benchmarks"][0]["task_limit"] = 1
    payload["benchmarks"][1]["task_limit"] = 1
    payload["sweep"]["benchmark_families"] = ["craft_mini"]
    payload["sweep"]["message_token_budgets"] = [0]
    payload["sweep"]["baselines"] = ["ma_ft"]
    temp_config = tmp_path / "zero-msg.yaml"
    temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    batch_index_path = run_batch(temp_config)
    batch_index = yaml.safe_load(batch_index_path.read_text(encoding="utf-8"))
    summary_path = Path(batch_index["trials"][0]["summary_path"])
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    assert summary["budget"]["mean_inter_agent_tokens"] == 0.0
    assert summary["derived"]["total_messages"] == 0


def test_maft_accounts_extra_fusion_calls_over_vote_local(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/p0a_clean_smoke.yaml").read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path)
    payload["benchmarks"][0]["task_limit"] = 1
    payload["benchmarks"][1]["task_limit"] = 1
    payload["sweep"]["benchmark_families"] = ["craft_mini"]
    payload["sweep"]["message_token_budgets"] = [32]
    payload["sweep"]["baselines"] = ["vote_local", "ma_ft"]
    temp_config = tmp_path / "compare.yaml"
    temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    batch_index_path = run_batch(temp_config)
    batch_index = yaml.safe_load(batch_index_path.read_text(encoding="utf-8"))
    summaries = {
        entry["baseline"]: yaml.safe_load(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        for entry in batch_index["trials"]
    }
    assert summaries["ma_ft"]["budget"]["task_model_call_counts"][0] > summaries["vote_local"]["budget"]["task_model_call_counts"][0]
    assert summaries["ma_ft"]["budget"]["task_inter_agent_tokens"][0] > 0
