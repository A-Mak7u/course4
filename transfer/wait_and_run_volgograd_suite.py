from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from volgograd_config import PROCESSED_ROOT, RAW_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT / "final_2013_2023_T_ERA5_LST_daynight.csv"
TARGET_CSV = PROCESSED_ROOT / "volgograd_target_daily_meteostat_2013_2023.csv"
ERA5_CSV = PROCESSED_ROOT / "volgograd_era5_daily_2013_2023.csv"
MODIS_CSV = PROCESSED_ROOT / "volgograd_modis_daily_2013_2023.csv"
FINAL_CSV = PROCESSED_ROOT / "volgograd_final_2013_2023_T_ERA5_LST_daynight.csv"
MODIS_RAW_DIR = RAW_ROOT / "modis_appeears"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Дождаться волгоградских артефактов и автоматически прогнать transfer-suite")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few-shot-stations", nargs="+", type=int, default=[5, 3])
    parser.add_argument("--output-root", default=None)
    return parser


def run_cmd(cmd: list[str]) -> None:
    print(f"[suite] run: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def maybe_parse_modis() -> bool:
    if MODIS_CSV.exists():
        return True
    result_files = list(MODIS_RAW_DIR.glob("*results.csv"))
    if not result_files:
        return False
    run_cmd(
        [
            str(PROJECT_ROOT / ".venv_geo/bin/python"),
            "transfer/parse_volgograd_modis_appeears.py",
            "--input-dir",
            str(MODIS_RAW_DIR),
            "--points-csv",
            str(MODIS_RAW_DIR / "points_used.csv"),
            "--output-csv",
            str(MODIS_CSV),
            "--coverage-csv",
            str(PROCESSED_ROOT / "volgograd_modis_coverage_2013_2023.csv"),
        ]
    )
    return MODIS_CSV.exists()


def maybe_build_final() -> bool:
    if FINAL_CSV.exists():
        return True
    if not TARGET_CSV.exists() or not ERA5_CSV.exists() or not MODIS_CSV.exists():
        return False
    run_cmd(
        [
            str(PROJECT_ROOT / ".venv_geo/bin/python"),
            "transfer/build_volgograd_final_dataset.py",
            "--target-csv",
            str(TARGET_CSV),
            "--era5-csv",
            str(ERA5_CSV),
            "--modis-csv",
            str(MODIS_CSV),
            "--output-csv",
            str(FINAL_CSV),
        ]
    )
    return FINAL_CSV.exists()


def run_transfer_case(
    *,
    output_root: Path,
    case_name: str,
    target_max_stations: int | None,
    args: argparse.Namespace,
) -> Path:
    outdir = output_root / case_name
    summary_path = outdir / "summary_metrics.csv"
    if summary_path.exists():
        print(f"[suite] skip existing case {case_name}: {summary_path}", flush=True)
        return outdir

    cmd = [
        str(PROJECT_ROOT / ".venv_geo/bin/python"),
        "transfer/xgb_transfer_experiment.py",
        "--source-csv",
        str(SOURCE_CSV),
        "--target-csv",
        str(FINAL_CSV),
        "--device",
        args.device,
        "--n-trials",
        str(args.n_trials),
        "--num-boost-round",
        str(args.num_boost_round),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(outdir),
    ]
    if target_max_stations is not None:
        cmd.extend(["--target-max-stations", str(target_max_stations)])
    run_cmd(cmd)
    return outdir


def build_suite_summary(output_root: Path, cases: list[tuple[str, Path]]) -> None:
    rows: list[dict[str, str | float | int]] = []
    for case_name, case_dir in cases:
        summary_path = case_dir / "summary_metrics.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["case"] = case_name
        rows.extend(df.to_dict("records"))

    if not rows:
        raise RuntimeError("Не удалось собрать suite_summary.csv: summary_metrics.csv не найдены")

    summary = pd.DataFrame(rows)
    summary = summary.rename(
        columns={
            "R2": "r2",
            "RMSE": "rmse",
            "MAE": "mae",
            "MedAE": "medae",
        }
    )
    summary_csv = output_root / "suite_summary.csv"
    summary.to_csv(summary_csv, index=False)

    plt.figure(figsize=(11, 6))
    plot_df = summary.copy()
    plot_df["label"] = plot_df["case"] + " / " + plot_df["mode"]
    order = plot_df.sort_values(["case", "mode"])["label"].tolist()
    plt.bar(order, plot_df.set_index("label").loc[order, "rmse"], color="#3a6ea5")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("RMSE")
    plt.title("Volgograd Transfer Suite")
    plt.tight_layout()
    plot_path = output_root / "suite_rmse.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    best = summary.sort_values("rmse").iloc[0].to_dict()
    meta = {
        "generated_at": dt.datetime.now().isoformat(),
        "best_case": best["case"],
        "best_mode": best["mode"],
        "best_rmse": float(best["rmse"]),
        "best_mae": float(best["mae"]),
        "cases": [name for name, _ in cases],
    }
    (output_root / "suite_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[suite] saved summary: {summary_csv}", flush=True)
    print(f"[suite] saved plot: {plot_path}", flush=True)


def main() -> None:
    args = make_parser().parse_args()
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "outputs_runs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_volgograd_transfer_suite"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[suite] output_root={output_root}", flush=True)
    print("[suite] waiting for ERA5+MODIS artifacts", flush=True)
    while True:
        modis_ready = maybe_parse_modis()
        final_ready = maybe_build_final() if modis_ready else False
        era5_ready = ERA5_CSV.exists()
        print(
            f"[suite] status era5_ready={era5_ready} modis_ready={modis_ready} final_ready={final_ready}",
            flush=True,
        )
        if final_ready:
            break
        time.sleep(args.poll_seconds)

    cases: list[tuple[str, Path]] = []
    cases.append(("full", run_transfer_case(output_root=output_root, case_name="full", target_max_stations=None, args=args)))
    for station_count in args.few_shot_stations:
        case_name = f"fewshot_{station_count}"
        case_dir = run_transfer_case(output_root=output_root, case_name=case_name, target_max_stations=station_count, args=args)
        cases.append((case_name, case_dir))

    build_suite_summary(output_root, cases)
    print("[suite] completed", flush=True)


if __name__ == "__main__":
    main()
