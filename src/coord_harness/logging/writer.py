from __future__ import annotations

import json
from pathlib import Path

from coord_harness.logging.schema import TrialSummary


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"

    def write_events(self, events: list[dict]) -> None:
        with self.events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def write_summary(self, summary: TrialSummary) -> None:
        self.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    def write_json(self, relative_path: str, payload: dict) -> Path:
        path = self.output_dir / relative_path
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
