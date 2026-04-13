from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сформировать transfer_modes_summary и таблицу относительно zero-shot"
    )
    parser.add_argument("--run-dir", required=True, help="Папка run xgb_transfer_experiment")
    parser.add_argument(
        "--input-csv-name",
        default="summary_metrics.csv",
        help="Имя исходного summary CSV внутри run-dir",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="transfer_modes_summary.csv",
    )
    parser.add_argument(
        "--output-vs-zeroshot-csv",
        default="transfer_modes_vs_zeroshot.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    src = run_dir / args.input_csv_name
    if not src.exists():
        raise FileNotFoundError(f"Missing summary metrics file: {src}")

    df = pd.read_csv(src).copy()
    if "mode" not in df.columns:
        raise RuntimeError(f"CSV does not contain 'mode' column: {src}")
    df["mode"] = df["mode"].astype(str)

    cols = [c for c in ["mode", "R2", "RMSE", "MAE", "MedAE", "n"] if c in df.columns]
    summary = df[cols].copy().sort_values("mode").reset_index(drop=True)

    out_summary = run_dir / args.output_summary_csv
    summary.to_csv(out_summary, index=False)

    if "zero-shot" not in set(summary["mode"].tolist()):
        raise RuntimeError("Mode 'zero-shot' not found; cannot build vs_zeroshot table")

    zero_row = summary[summary["mode"] == "zero-shot"].iloc[0]
    rows: list[dict[str, float | int | str]] = []
    for _, row in summary.iterrows():
        if str(row["mode"]) == "zero-shot":
            continue
        payload: dict[str, float | int | str] = {"mode": str(row["mode"])}
        if "RMSE" in summary.columns:
            payload["RMSE_gain_vs_zeroshot"] = float(zero_row["RMSE"] - row["RMSE"])
        if "MAE" in summary.columns:
            payload["MAE_gain_vs_zeroshot"] = float(zero_row["MAE"] - row["MAE"])
        if "R2" in summary.columns:
            payload["R2_gain_vs_zeroshot"] = float(row["R2"] - zero_row["R2"])
        rows.append(payload)
    vs_df = pd.DataFrame(rows)
    out_vs = run_dir / args.output_vs_zeroshot_csv
    vs_df.to_csv(out_vs, index=False)

    print(f"Saved transfer summary: {out_summary}")
    print(f"Saved transfer vs-zero-shot: {out_vs}")


if __name__ == "__main__":
    main()
