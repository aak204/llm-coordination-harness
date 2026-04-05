from __future__ import annotations

from pathlib import Path

import yaml

from coord_harness.runner.executor import run_batch


def test_smoke_run_produces_batch_index_and_summaries(tmp_path: Path) -> None:
    config_path = Path("archive/configs/p0a_clean_smoke.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path)
    temp_config = tmp_path / "smoke.yaml"
    temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    batch_index_path = run_batch(temp_config)
    assert batch_index_path.exists()

    batch_index = yaml.safe_load(batch_index_path.read_text(encoding="utf-8"))
    assert len(batch_index["trials"]) == 6
    ma_ft_entries = [entry for entry in batch_index["trials"] if entry["baseline"] == "ma_ft"]
    assert ma_ft_entries

    summary_path = Path(ma_ft_entries[0]["summary_path"])
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    assert summary["run"]["mode"] == "research_strict"
    assert summary["derived"]["recomputed_offline"] is True
    assert summary["outcomes"]["regime"] in {"help", "saturation", "collapse", None}
