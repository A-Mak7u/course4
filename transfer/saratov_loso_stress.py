from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (
    TARGET_COLUMN,
    build_feature_frame,
    choose_validation_year,
    compute_metrics,
    ensure_dir,
    evaluate_model,
    load_dataset,
    resolve_feature_list,
    save_json,
    split_by_year,
    train_xgb,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOSO stress-test для саратовской ветки (unseen station on test)")
    parser.add_argument("--input-csv", default="final_2013_2023_T_ERA5_LST_daynight.csv")
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument(
        "--params-json",
        default="outputs_runs/20250916_171729_lags123_spatial/params.json",
        help="Базовые параметры XGBoost; если файла нет, используются встроенные",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-boost-round", type=int, default=2500)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--min-test-rows", type=int, default=120)
    parser.add_argument("--max-stations", type=int, default=None, help="Ограничить число станций для быстрого прогона")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def load_params(path: Path, device: str, seed: int) -> dict[str, float | int]:
    if path.exists():
        params = pd.read_json(path, typ="series").to_dict()
    else:
        params = {
            "max_depth": 10,
            "learning_rate": 0.011,
            "subsample": 0.60,
            "colsample_bytree": 0.94,
            "reg_lambda": 0.011,
            "alpha": 0.029,
            "min_child_weight": 12,
        }
    params.update(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "seed": seed,
        }
    )
    return params


def plot_rmse_by_station(df: pd.DataFrame, output_png: Path) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("RMSE", ascending=False)
    plt.figure(figsize=(11, 4.5))
    plt.bar(plot_df["station"].astype(str), plot_df["RMSE"])
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("RMSE")
    plt.title("LOSO test RMSE by held-out station")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def to_celsius_if_kelvin(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    med = np.nanmedian(arr)
    if np.isfinite(med) and med > 150:
        return arr - 273.15
    return arr


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ensure_dir(args.output_dir or f"outputs_runs/{ts}_saratov_loso_stress")

    df, meta = load_dataset(args.input_csv)
    df = build_feature_frame(
        df,
        meta,
        train_mask=None,
        zero_inflated_precip=False,
        include_station_mean=False,
    )
    features = resolve_feature_list(df, include_station_mean=False)

    train_mask, test_mask = split_by_year(
        df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    test_df = df.loc[test_mask].dropna(subset=[TARGET_COLUMN]).copy()
    station_col = meta.station_col

    station_counts = test_df[station_col].value_counts().rename_axis("station").reset_index(name="n_test")
    station_counts = station_counts[station_counts["n_test"] >= args.min_test_rows].copy()
    if station_counts.empty:
        raise RuntimeError("Нет станций, удовлетворяющих min-test-rows")
    if args.max_stations is not None and len(station_counts) > args.max_stations:
        station_counts = station_counts.sample(args.max_stations, random_state=args.seed).sort_values("station")

    params = load_params(Path(args.params_json), device=args.device, seed=args.seed)
    rows: list[dict[str, float | int | str]] = []

    print(
        f"[loso] start stations={len(station_counts)} features={len(features)} num_boost_round={args.num_boost_round}",
        flush=True,
    )

    for i, station in enumerate(station_counts["station"].tolist(), start=1):
        train_station_mask = train_mask & (df[station_col] != station)
        train_df = df.loc[train_station_mask].dropna(subset=[TARGET_COLUMN]).copy()
        test_station_mask = test_mask & (df[station_col] == station)
        test_station_df = df.loc[test_station_mask].dropna(subset=[TARGET_COLUMN]).copy()
        if train_df.empty or test_station_df.empty:
            continue

        val_year = choose_validation_year(train_df)
        inner_train = train_df[train_df["year"] < val_year].copy()
        inner_val = train_df[train_df["year"] == val_year].copy()
        if inner_train.empty or inner_val.empty:
            continue

        print(
            f"[loso] station={station} ({i}/{len(station_counts)}) train_rows={len(train_df)} test_rows={len(test_station_df)}",
            flush=True,
        )
        model = train_xgb(
            train_df,
            inner_val,
            features,
            params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            progress_label=f"loso_{station}",
        )
        preds, metrics = evaluate_model(model, test_station_df, features)
        row: dict[str, float | int | str] = {"station": station, **metrics}

        if "Temperature_2m" in test_station_df.columns:
            baseline_df = test_station_df[[TARGET_COLUMN, "Temperature_2m"]].dropna().copy()
            if not baseline_df.empty:
                t2m_baseline = to_celsius_if_kelvin(baseline_df["Temperature_2m"])
                baseline = compute_metrics(baseline_df[TARGET_COLUMN], t2m_baseline)
                row["baseline_rmse_t2m"] = baseline["RMSE"]  # type: ignore[index]
                row["baseline_mae_t2m"] = baseline["MAE"]  # type: ignore[index]
                row["rmse_gain_vs_t2m"] = float(baseline["RMSE"]) - float(metrics["RMSE"])  # type: ignore[index]
                row["mae_gain_vs_t2m"] = float(baseline["MAE"]) - float(metrics["MAE"])  # type: ignore[index]

        rows.append(row)

    if not rows:
        raise RuntimeError("LOSO не дал ни одной валидной станции")

    by_station = pd.DataFrame(rows).sort_values("RMSE", ascending=False)
    by_station.to_csv(Path(outdir) / "metrics_by_station_loso.csv", index=False)
    plot_rmse_by_station(by_station, Path(outdir) / "rmse_by_station_loso.png")

    summary = {
        "stations_evaluated": int(len(by_station)),
        "RMSE_mean": float(by_station["RMSE"].mean()),
        "RMSE_median": float(by_station["RMSE"].median()),
        "RMSE_p90": float(np.quantile(by_station["RMSE"], 0.90)),
        "MAE_mean": float(by_station["MAE"].mean()),
        "worst_station_by_rmse": str(by_station.iloc[0]["station"]),
        "worst_station_rmse": float(by_station.iloc[0]["RMSE"]),
        "best_station_by_rmse": str(by_station.iloc[-1]["station"]),
        "best_station_rmse": float(by_station.iloc[-1]["RMSE"]),
        "input_csv": args.input_csv,
        "feature_count": int(len(features)),
        "params_json": args.params_json,
    }
    save_json(Path(outdir) / "summary_metrics.json", summary)
    save_json(Path(outdir) / "features_used.json", features)
    save_json(Path(outdir) / "params_used.json", params)

    print(f"Saved LOSO stress run: {outdir}")


if __name__ == "__main__":
    main()
