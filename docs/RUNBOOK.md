# Runbook

## Install

```powershell
pip install -e .[dev]
```

## Release Configs

Public release state keeps only these live configs in [configs](d:/Dev/TEST_HYPO/configs):

- [p0a_calibrated_full_live.yaml](d:/Dev/TEST_HYPO/configs/p0a_calibrated_full_live.yaml)
- [p0b_attacks_live.yaml](d:/Dev/TEST_HYPO/configs/p0b_attacks_live.yaml)
- [RELEASE_FREEZE.json](d:/Dev/TEST_HYPO/configs/RELEASE_FREEZE.json)

Historical exploratory configs were moved to [archive/configs](d:/Dev/TEST_HYPO/archive/configs).

## Validate

```powershell
$env:PYTHONPATH = "src"
python -m coord_harness.cli validate-config configs/p0a_calibrated_full_live.yaml
python -m coord_harness.cli validate-config configs/p0b_attacks_live.yaml
```

## Run Clean

```powershell
$env:PYTHONPATH = "src"
$env:OPENROUTER_API_KEY = "<key>"
python -m coord_harness.cli run configs/p0a_calibrated_full_live.yaml
python -m coord_harness.cli evaluate-gate outputs/p0a-calibrated-full-live --primary-target help_vs_rest
python -m coord_harness.cli analyze-predictor outputs/p0a-calibrated-full-live
```

## Run Stress

```powershell
$env:PYTHONPATH = "src"
$env:OPENROUTER_API_KEY = "<key>"
python -m coord_harness.cli run configs/p0b_attacks_live.yaml
```

## Plot

```powershell
$env:PYTHONPATH = "src"
python -m coord_harness.analysis.plotter
```

Figures are written to [docs/figures](d:/Dev/TEST_HYPO/docs/figures).

## Golden Outputs

Public release state keeps only these experiment directories in [outputs](d:/Dev/TEST_HYPO/outputs):

- [p0a-calibrated-full-live](d:/Dev/TEST_HYPO/outputs/p0a-calibrated-full-live)
- [p0b-attacks-live](d:/Dev/TEST_HYPO/outputs/p0b-attacks-live)

## OpenRouter Discipline

Research runs use `research_strict` only:

- exact model ID pinning
- explicit provider pinning
- no `openrouter/auto`
- no provider fallback
- route / pricing / snapshot logging

Batch execution is parallelized by independent trial cell with `run.max_concurrency`.
The release configs use `25`.
