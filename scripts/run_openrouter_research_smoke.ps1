$env:PYTHONPATH = "src"
python -m coord_harness.cli validate-config configs/openrouter_research_smoke.yaml
python -m coord_harness.cli run configs/openrouter_research_smoke.yaml
