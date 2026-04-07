from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_SCRIPT = Path(__file__).resolve().with_name("xgb_transfer_experiment.py")
DEFAULT_INPUT_CSV = PROJECT_ROOT / "final_2013_2023_T_ERA5_LST_daynight.csv"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight-проверка transfer-пайплайна на spatial holdout")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--station-col", default="Cod")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--target-col", default="T")
    parser.add_argument("--lon-col", default="X_final")
    parser.add_argument("--lat-col", default="Y_final")
    parser.add_argument("--target-max-stations-grid", nargs="+", type=int, default=[0, 5, 3])
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["west_to_east", "east_to_west"],
        choices=["west_to_east", "east_to_west"],
        help="Какие направления spatial-transfer запускать",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero-shot", "finetune", "scratch"],
        choices=["zero-shot", "finetune", "scratch"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--num-boost-round", type=int, default=2500)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-inflated-precip", action="store_true")
    parser.add_argument("--winter-only", action="store_true")
    parser.add_argument("--post-bias-correction", action="store_true")
    parser.add_argument("--keep-case-artifacts", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def build_spatial_station_split(
    df: pd.DataFrame,
    *,
    station_col: str,
    lon_col: str,
    lat_col: str,
) -> tuple[pd.DataFrame, float]:
    stations = (
        df[[station_col, lon_col, lat_col]]
        .dropna(subset=[lon_col, lat_col])
        .drop_duplicates(subset=[station_col])
        .copy()
        .sort_values([lon_col, lat_col, station_col])
        .reset_index(drop=True)
    )
    if stations.empty:
        raise RuntimeError("Не удалось получить непустой список станций с координатами")

    lon_median = float(stations[lon_col].median())
    stations["region"] = stations[lon_col].ge(lon_median).map({True: "east", False: "west"})
    region_counts = stations["region"].value_counts()
    if len(region_counts) != 2:
        raise RuntimeError("Spatial split выродился: не удалось получить две стороны")
    return stations, lon_median


def write_region_csvs(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    station_col: str,
    temp_dir: Path,
) -> dict[str, Path]:
    tagged = df.merge(stations[[station_col, "region"]], on=station_col, how="inner")
    paths: dict[str, Path] = {}
    for region in ("west", "east"):
        path = temp_dir / f"saratov_{region}.csv"
        tagged.loc[tagged["region"] == region].drop(columns=["region"]).to_csv(path, index=False)
        paths[region] = path
    return paths


def write_low_station_target_csv(
    df: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    station_col: str,
    date_col: str,
    target_col: str,
    target_region: str,
    target_max_stations: int,
    train_end_year: int,
    seed: int,
    temp_dir: Path,
) -> tuple[Path, int, list[int | str]]:
    pool = stations[(stations["region"] == target_region) & (stations["has_target"] == 1)].copy()
    if pool.empty:
        raise RuntimeError(f"Для региона {target_region} нет станций с наблюдаемой T")

    actual_budget = int(min(target_max_stations, len(pool)))
    sampled = pool.sort_values(station_col).sample(actual_budget, random_state=seed)
    station_ids = sampled[station_col].tolist()
    region_station_ids = stations.loc[stations["region"] == target_region, station_col].tolist()
    target_df = df[df[station_col].isin(region_station_ids)].copy()
    target_df[date_col] = pd.to_datetime(target_df[date_col])

    mask_train_hidden = (target_df[date_col].dt.year <= train_end_year) & (~target_df[station_col].isin(station_ids))
    target_df.loc[mask_train_hidden, target_col] = pd.NA

    path = temp_dir / f"saratov_{target_region}_fewtrain_n{actual_budget:02d}.csv"
    target_df.to_csv(path, index=False)
    return path, actual_budget, station_ids


def run_transfer_case(
    *,
    source_csv: Path,
    target_csv: Path,
    outdir: Path,
    modes: list[str],
    device: str,
    n_trials: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    zero_inflated_precip: bool,
    winter_only: bool,
    post_bias_correction: bool,
) -> None:
    cmd = [
        sys.executable,
        str(TRANSFER_SCRIPT),
        "--source-csv",
        str(source_csv),
        "--target-csv",
        str(target_csv),
        "--device",
        device,
        "--n-trials",
        str(n_trials),
        "--num-boost-round",
        str(num_boost_round),
        "--early-stopping-rounds",
        str(early_stopping_rounds),
        "--seed",
        str(seed),
        "--output-dir",
        str(outdir),
        "--train-start-year",
        str(train_start_year),
        "--train-end-year",
        str(train_end_year),
        "--test-start-year",
        str(test_start_year),
        "--test-end-year",
        str(test_end_year),
        "--modes",
        *modes,
    ]
    if zero_inflated_precip:
        cmd.append("--zero-inflated-precip")
    if winter_only:
        cmd.append("--winter-only")
    if post_bias_correction:
        cmd.append("--post-bias-correction")

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def collect_case_summary(
    *,
    run_dir: Path,
    direction: str,
    source_region: str,
    target_region: str,
    requested_budget: int,
) -> list[dict[str, float | int | str]]:
    summary_path = run_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Не найден summary_metrics.csv в {run_dir}")

    summary_df = pd.read_csv(summary_path)
    summary_bias_path = run_dir / "summary_metrics_bias_corrected.csv"
    summary_bias_df = pd.read_csv(summary_bias_path) if summary_bias_path.exists() else pd.DataFrame()
    bias_by_mode: dict[str, dict[str, float | int | str]] = {}
    if not summary_bias_df.empty:
        for _, b_row in summary_bias_df.iterrows():
            bias_by_mode[str(b_row["mode"])] = {
                "R2": float(b_row["R2"]),
                "RMSE": float(b_row["RMSE"]),
                "MAE": float(b_row["MAE"]),
                "MedAE": float(b_row["MedAE"]),
                "n": int(b_row["n"]),
            }
    rows: list[dict[str, float | int | str]] = []
    run_dir_resolved = run_dir.resolve()
    project_root_resolved = PROJECT_ROOT.resolve()
    try:
        run_dir_label = str(run_dir_resolved.relative_to(project_root_resolved))
    except ValueError:
        run_dir_label = str(run_dir_resolved)

    for _, row in summary_df.iterrows():
        mode_dir = run_dir / str(row["mode"]).replace("-", "_")
        station_metrics_path = mode_dir / "metrics_by_station_test.csv"
        target_station_count = None
        if station_metrics_path.exists():
            target_station_count = int(len(pd.read_csv(station_metrics_path)))

        rows.append(
            {
                "direction": direction,
                "source_region": source_region,
                "target_region": target_region,
                "requested_target_stations": int(requested_budget),
                "target_station_count": target_station_count if target_station_count is not None else int(requested_budget),
                "mode": str(row["mode"]),
                "R2": float(row["R2"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "MedAE": float(row["MedAE"]),
                "n": int(row["n"]),
                "run_dir": run_dir_label,
            }
        )
        mode_name = str(row["mode"])
        if mode_name in bias_by_mode:
            b = bias_by_mode[mode_name]
            rows.append(
                {
                    "direction": direction,
                    "source_region": source_region,
                    "target_region": target_region,
                    "requested_target_stations": int(requested_budget),
                    "target_station_count": target_station_count if target_station_count is not None else int(requested_budget),
                    "mode": f"{mode_name}+bias",
                    "R2": float(b["R2"]),
                    "RMSE": float(b["RMSE"]),
                    "MAE": float(b["MAE"]),
                    "MedAE": float(b["MedAE"]),
                    "n": int(b["n"]),
                    "run_dir": run_dir_label,
                }
            )
    return rows


def save_metric_plot(summary_df: pd.DataFrame, metric: str, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, direction in zip(axes, sorted(summary_df["direction"].unique())):
        sub = summary_df[summary_df["direction"] == direction].copy()
        for mode in ["zero-shot", "finetune", "scratch"]:
            mode_df = sub[sub["mode"] == mode].sort_values("target_station_count")
            if mode_df.empty:
                continue
            ax.plot(
                mode_df["target_station_count"].to_numpy(),
                mode_df[metric].to_numpy(),
                marker="o",
                linewidth=2,
                label=mode,
            )
        ax.set_title(direction)
        ax.set_xlabel("Target stations")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(metric)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def prune_case_artifacts(run_dir: Path) -> None:
    for child in run_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir or PROJECT_ROOT / "outputs_runs" / f"{ts}_spatial_transfer_preflight")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    stations, lon_median = build_spatial_station_split(
        df,
        station_col=args.station_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
    )
    active_station_ids = set(df.loc[df["T"].notna(), args.station_col].dropna().unique().tolist())
    stations["has_target"] = stations[args.station_col].isin(active_station_ids).astype(int)

    stations.sort_values(["region", args.lon_col, args.lat_col, args.station_col]).to_csv(
        outdir / "station_split.csv",
        index=False,
    )

    split_meta = {
        "input_csv": str(Path(args.input_csv)),
        "station_col": args.station_col,
        "date_col": args.date_col,
        "target_col": args.target_col,
        "lon_col": args.lon_col,
        "lat_col": args.lat_col,
        "lon_median": lon_median,
        "station_counts": stations["region"].value_counts().sort_index().to_dict(),
        "active_station_counts": stations.groupby("region")["has_target"].sum().astype(int).to_dict(),
        "low_station_protocol": "full target test, masked target-train T outside sampled calibration stations",
        "train_years": [args.train_start_year, args.train_end_year],
        "test_years": [args.test_start_year, args.test_end_year],
        "target_max_stations_grid": args.target_max_stations_grid,
        "directions": args.directions,
        "modes": args.modes,
        "device": args.device,
        "n_trials": args.n_trials,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "seed": args.seed,
        "zero_inflated_precip": args.zero_inflated_precip,
        "winter_only": args.winter_only,
        "post_bias_correction": args.post_bias_correction,
        "keep_case_artifacts": args.keep_case_artifacts,
    }
    (outdir / "split_meta.json").write_text(json.dumps(split_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    station_counts = stations["region"].value_counts().to_dict()
    full_budget = int(min(station_counts.values()))

    summary_rows: list[dict[str, float | int | str]] = []
    direction_map = {
        "west_to_east": ("west", "east"),
        "east_to_west": ("east", "west"),
    }
    directions = [direction_map[key] for key in args.directions]

    with tempfile.TemporaryDirectory(prefix="spatial_transfer_") as temp_root:
        temp_root_path = Path(temp_root)
        region_paths = write_region_csvs(
            df,
            stations,
            station_col=args.station_col,
            temp_dir=temp_root_path,
        )

        for source_region, target_region in directions:
            direction_name = f"{source_region}_to_{target_region}"
            for raw_budget in args.target_max_stations_grid:
                if raw_budget <= 0:
                    target_csv = region_paths[target_region]
                    requested_budget = full_budget
                    sampled_station_ids: list[int | str] | None = None
                    run_name = f"{direction_name}__all"
                else:
                    target_csv, requested_budget, sampled_station_ids = write_low_station_target_csv(
                        df,
                        stations,
                        station_col=args.station_col,
                        date_col=args.date_col,
                        target_col=args.target_col,
                        target_region=target_region,
                        target_max_stations=int(raw_budget),
                        train_end_year=args.train_end_year,
                        seed=args.seed,
                        temp_dir=temp_root_path,
                    )
                    run_name = f"{direction_name}__fewtrain{requested_budget:02d}"
                run_dir = outdir / run_name

                if sampled_station_ids is not None:
                    (run_dir / "sampled_target_stations.json").parent.mkdir(parents=True, exist_ok=True)
                    (run_dir / "sampled_target_stations.json").write_text(
                        json.dumps(sampled_station_ids, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                run_transfer_case(
                    source_csv=region_paths[source_region],
                    target_csv=target_csv,
                    outdir=run_dir,
                    modes=args.modes,
                    device=args.device,
                    n_trials=args.n_trials,
                    num_boost_round=args.num_boost_round,
                    early_stopping_rounds=args.early_stopping_rounds,
                    seed=args.seed,
                    train_start_year=args.train_start_year,
                    train_end_year=args.train_end_year,
                    test_start_year=args.test_start_year,
                    test_end_year=args.test_end_year,
                    zero_inflated_precip=args.zero_inflated_precip,
                    winter_only=args.winter_only,
                    post_bias_correction=args.post_bias_correction,
                )
                summary_rows.extend(
                    collect_case_summary(
                        run_dir=run_dir,
                        direction=direction_name,
                        source_region=source_region,
                        target_region=target_region,
                        requested_budget=requested_budget,
                    )
                )
                if not args.keep_case_artifacts:
                    prune_case_artifacts(run_dir)

    summary_df = pd.DataFrame(summary_rows).sort_values(["direction", "target_station_count", "mode"]).reset_index(drop=True)
    summary_df.to_csv(outdir / "summary_all_cases.csv", index=False)

    save_metric_plot(summary_df, "RMSE", outdir / "summary_rmse.png")
    save_metric_plot(summary_df, "MAE", outdir / "summary_mae.png")

    best_by_case = (
        summary_df.sort_values(["direction", "target_station_count", "RMSE"])
        .groupby(["direction", "target_station_count"], as_index=False)
        .first()
    )
    best_by_case.to_csv(outdir / "best_mode_by_case.csv", index=False)

    print(f"Saved spatial transfer preflight: {outdir}")


if __name__ == "__main__":
    main()
