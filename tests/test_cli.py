from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_python_module_cli_executes_validate_config() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-m", "coord_harness.cli", "validate-config", "configs/p0a_clean_smoke.yaml"],
        cwd=Path("."),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"trial_count": 6' in result.stdout
