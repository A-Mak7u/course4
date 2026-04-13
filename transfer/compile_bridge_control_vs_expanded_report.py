from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сбор единого отчёта control vs expanded по bridge-рандам"
    )
    parser.add_argument("--control-run-dir", required=True)
    parser.add_argument("--expanded-run-dir", required=True)
    parser.add_argument(
        "--output-csv",
        default="outputs_runs/bridge_control_vs_expanded_report.csv",
    )
    return parser.parse_args()


def load_one(run_dir: Path, dataset_tag: str) -> pd.DataFrame:
    metrics_path = run_dir / "variant_metrics.csv"
    risk_path = run_dir / "station_risk_summary_test.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing file: {metrics_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing file: {risk_path}")

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["split"] == "test"].copy()
    keep_cols = [c for c in ["variant", "R2", "RMSE", "MAE", "MedAE", "n"] if c in metrics.columns]
    metrics = metrics[keep_cols].copy()

    risk = pd.read_csv(risk_path).copy()
    risk_keep = [
        c
        for c in [
            "variant",
            "stations_total",
            "improved_station_count",
            "worsened_station_count",
            "mean_mae_gain",
            "median_mae_gain",
        ]
        if c in risk.columns
    ]
    risk = risk[risk_keep].copy()

    merged = metrics.merge(risk, on="variant", how="left")
    merged.insert(0, "dataset", dataset_tag)
    return merged


def main() -> None:
    args = parse_args()
    control_dir = Path(args.control_run_dir)
    expanded_dir = Path(args.expanded_run_dir)

    control = load_one(control_dir, "control_selected125_2013_2023")
    expanded = load_one(expanded_dir, "expanded_min10_2013_2023")

    report = pd.concat([control, expanded], ignore_index=True)
    report = report.sort_values(["variant", "dataset"]).reset_index(drop=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"Saved control-vs-expanded report: {out}")


if __name__ == "__main__":
    main()
