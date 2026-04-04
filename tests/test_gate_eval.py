from __future__ import annotations

from pathlib import Path

import yaml

from coord_harness.evaluation.gate import write_gate_report
from coord_harness.runner.executor import run_batch


def test_gate_report_writes_after_mock_pilot(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/p0a_clean_pilot_mock.yaml").read_text(encoding="utf-8"))
    payload["run"]["output_root"] = str(tmp_path)
    temp_config = tmp_path / "pilot.yaml"
    temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    batch_index_path = run_batch(temp_config)
    report_path = write_gate_report(batch_index_path, primary_target="help_vs_rest")
    assert report_path.exists()
