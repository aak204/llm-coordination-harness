from __future__ import annotations

import argparse
from pathlib import Path

from coord_harness.analysis.replay import recompute_experiment_deriveds
from coord_harness.evaluation.gate import write_gate_report
from coord_harness.evaluation.predictor_analysis import write_predictor_analysis
from coord_harness.runner.executor import run_batch, validate_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal multi-agent coordination research harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a batch config.")
    validate_parser.add_argument("config", type=Path)

    run_parser = subparsers.add_parser("run", help="Run a batch config.")
    run_parser.add_argument("config", type=Path)

    gate_parser = subparsers.add_parser("evaluate-gate", help="Evaluate held-out family gate metrics.")
    gate_parser.add_argument("experiment", type=Path)
    gate_parser.add_argument("--primary-target", default="help_vs_rest")

    replay_parser = subparsers.add_parser("recompute-derived", help="Recompute derived variables offline from logs.")
    replay_parser.add_argument("experiment", type=Path)

    predictor_parser = subparsers.add_parser("analyze-predictor", help="Offline predictor analysis on existing artifacts.")
    predictor_parser.add_argument("experiment", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate-config":
        print(validate_only(args.config))
        return
    if args.command == "run":
        print(run_batch(args.config))
        return
    if args.command == "evaluate-gate":
        print(write_gate_report(args.experiment, primary_target=args.primary_target))
        return
    if args.command == "recompute-derived":
        print(recompute_experiment_deriveds(args.experiment))
        return
    if args.command == "analyze-predictor":
        print(write_predictor_analysis(args.experiment))
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
