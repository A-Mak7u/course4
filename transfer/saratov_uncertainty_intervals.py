from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline_common import (
    TARGET_COLUMN,
    apply_station_train_mean_from_reference,
    build_feature_frame,
    choose_validation_year,
    ensure_dir,
    load_dataset,
    resolve_feature_list,
    save_json,
    split_by_year,
    train_xgb,
    tune_xgb,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Интервалы неопределенности (P10/P50/P90) для саратовской базы")
    parser.add_argument("--input-csv", default="final_2013_2023_T_ERA5_LST_daynight.csv")
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-boost-round", type=int, default=2500)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument(
        "--calibration-method",
        choices=["scale", "conformal_additive", "conformal_monthly"],
        default="scale",
        help="Способ калибровки интервала на inner_val",
    )
    parser.add_argument(
        "--monthly-calib-min-samples",
        type=int,
        default=120,
        help="Минимум наблюдений месяца для отдельной monthly conformal-калибровки",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def compute_interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    p50: np.ndarray,
    upper: np.ndarray,
    *,
    crossing_rate_raw: float | None = None,
) -> dict[str, float | int]:
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return {
        "n": int(len(y_true)),
        "coverage_p10_p90": float(inside.mean()),
        "interval_width_mean": float(width.mean()),
        "interval_width_median": float(np.median(width)),
        "interval_width_p90": float(np.quantile(width, 0.90)),
        "crossing_rate_raw": float(crossing_rate_raw) if crossing_rate_raw is not None else float("nan"),
        "mae_p50": float(np.mean(np.abs(y_true - p50))),
        "rmse_p50": float(np.sqrt(np.mean((y_true - p50) ** 2))),
    }


def save_group_interval_metrics(
    df: pd.DataFrame,
    group_col: str,
    output_csv: Path,
) -> None:
    rows: list[dict[str, float | int | str]] = []
    for group, group_df in df.groupby(group_col):
        if len(group_df) < 5:
            continue
        rows.append(
            {
                "group": group,
                "n": int(len(group_df)),
                "coverage_p10_p90": float(group_df["inside"].mean()),
                "interval_width_mean": float(group_df["interval_width"].mean()),
                "interval_width_median": float(group_df["interval_width"].median()),
                "mae_p50": float(np.mean(np.abs(group_df[TARGET_COLUMN] - group_df["y_pred_p50"]))),
                "rmse_p50": float(np.sqrt(np.mean((group_df[TARGET_COLUMN] - group_df["y_pred_p50"]) ** 2))),
            }
        )
    if not rows:
        pd.DataFrame(
            columns=[
                "group",
                "n",
                "coverage_p10_p90",
                "interval_width_mean",
                "interval_width_median",
                "mae_p50",
                "rmse_p50",
            ]
        ).to_csv(output_csv, index=False)
        return
    pd.DataFrame(rows).sort_values("group").to_csv(output_csv, index=False)


def plot_interval_width_hist(width: np.ndarray, output_png: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(width, bins=60)
    plt.xlabel("P90 - P10")
    plt.ylabel("Count")
    plt.title("Interval width histogram (test)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def plot_monthly_coverage(monthly_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(monthly_csv)
    if df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df["group"], df["coverage_p10_p90"], marker="o")
    plt.axhline(0.80, color="r", linestyle="--", linewidth=1.2, label="target 0.80")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Month")
    plt.ylabel("Coverage")
    plt.title("Coverage by month (P10-P90)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def build_interval(
    p10_raw: np.ndarray,
    p50: np.ndarray,
    p90_raw: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    raw_lower = np.minimum(p10_raw, p90_raw)
    raw_upper = np.maximum(p10_raw, p90_raw)
    left_half = np.maximum(p50 - raw_lower, 1e-6)
    right_half = np.maximum(raw_upper - p50, 1e-6)
    lower = p50 - scale * left_half
    upper = p50 + scale * right_half
    return lower, upper


def estimate_calibration_scale(
    y_true: np.ndarray,
    p10_raw: np.ndarray,
    p50: np.ndarray,
    p90_raw: np.ndarray,
    target_coverage: float,
) -> float:
    raw_lower = np.minimum(p10_raw, p90_raw)
    raw_upper = np.maximum(p10_raw, p90_raw)
    left_half = np.maximum(p50 - raw_lower, 1e-6)
    right_half = np.maximum(raw_upper - p50, 1e-6)

    errors = np.abs(y_true - p50)
    denom = np.where(y_true <= p50, left_half, right_half)
    ratio = errors / np.maximum(denom, 1e-6)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return 1.0

    q = float(np.quantile(ratio, np.clip(target_coverage, 0.01, 0.99)))
    # keep calibration bounded to avoid pathological widening/shrinking
    return float(np.clip(q, 0.5, 3.0))


def estimate_conformal_delta(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> float:
    scores = np.maximum(lower - y_true, y_true - upper)
    scores = np.maximum(scores, 0.0)
    if len(scores) == 0:
        return 0.0
    q = float(np.quantile(scores, np.clip(target_coverage, 0.01, 0.999), method="higher"))
    return max(0.0, q)


def estimate_monthly_conformal_deltas(
    calib_df: pd.DataFrame,
    target_coverage: float,
    min_samples: int,
) -> tuple[dict[int, float], float]:
    global_delta = estimate_conformal_delta(
        calib_df["y_true"].to_numpy(),
        calib_df["lower_raw"].to_numpy(),
        calib_df["upper_raw"].to_numpy(),
        target_coverage=target_coverage,
    )

    month_to_delta: dict[int, float] = {}
    for month, month_df in calib_df.groupby("month"):
        if len(month_df) < min_samples:
            continue
        month_to_delta[int(month)] = estimate_conformal_delta(
            month_df["y_true"].to_numpy(),
            month_df["lower_raw"].to_numpy(),
            month_df["upper_raw"].to_numpy(),
            target_coverage=target_coverage,
        )
    return month_to_delta, float(global_delta)


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(ensure_dir(args.output_dir or f"outputs_runs/{ts}_saratov_uncertainty"))

    df, meta = load_dataset(args.input_csv)
    train_mask0, test_mask0 = split_by_year(
        df.assign(year=pd.to_datetime(df[meta.date_col]).dt.year),
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )

    df = build_feature_frame(
        df,
        meta,
        train_mask=train_mask0,
        zero_inflated_precip=False,
        include_station_mean=False,
    )
    station_col = meta.station_col

    train_mask, test_mask = split_by_year(
        df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    train = df.loc[train_mask].dropna(subset=[TARGET_COLUMN]).copy()
    test = df.loc[test_mask].dropna(subset=[TARGET_COLUMN]).copy()
    if train.empty or test.empty:
        raise RuntimeError("После split train/test получились пустые выборки")

    val_year = choose_validation_year(train)
    inner_train = train[train["year"] < val_year].copy()
    inner_val = train[train["year"] == val_year].copy()
    if inner_train.empty or inner_val.empty:
        raise RuntimeError("Не удалось сформировать inner_train/inner_val")

    inner_train = apply_station_train_mean_from_reference(inner_train, inner_train, meta)
    inner_val = apply_station_train_mean_from_reference(inner_val, inner_train, meta)
    train = apply_station_train_mean_from_reference(train, train, meta)
    test = apply_station_train_mean_from_reference(test, train, meta)
    features = resolve_feature_list(train, include_station_mean=True)

    if args.calibration_method == "scale":
        tune_train = inner_train
        tune_val = inner_val
        fit_train = train
        fit_eval = inner_val
        calib_df = inner_val
        calibration_split_meta = {
            "strategy": "legacy_scale",
            "tune_train_years": [int(tune_train["year"].min()), int(tune_train["year"].max())],
            "tune_val_years": [int(tune_val["year"].min()), int(tune_val["year"].max())],
            "fit_train_years": [int(fit_train["year"].min()), int(fit_train["year"].max())],
            "calib_years": [int(calib_df["year"].min()), int(calib_df["year"].max())],
        }
    else:
        tune_val_year = choose_validation_year(inner_train)
        tune_train = inner_train[inner_train["year"] < tune_val_year].copy()
        tune_val = inner_train[inner_train["year"] == tune_val_year].copy()
        if tune_train.empty or tune_val.empty:
            raise RuntimeError("Не удалось сформировать split-conformal tune_train/tune_val")

        fit_train = inner_train
        fit_eval = tune_val
        calib_df = inner_val
        calibration_split_meta = {
            "strategy": "split_conformal_holdout",
            "tune_train_years": [int(tune_train["year"].min()), int(tune_train["year"].max())],
            "tune_val_years": [int(tune_val["year"].min()), int(tune_val["year"].max())],
            "fit_train_years": [int(fit_train["year"].min()), int(fit_train["year"].max())],
            "calib_years": [int(calib_df["year"].min()), int(calib_df["year"].max())],
        }

    base_params = tune_xgb(
        tune_train,
        tune_val,
        features,
        device=args.device,
        n_trials=args.n_trials,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        progress_label="uncertainty_tune",
    )

    p50_params = dict(base_params)
    p50_params["objective"] = "reg:squarederror"
    p10_params = dict(base_params)
    p10_params["objective"] = "reg:quantileerror"
    p10_params["quantile_alpha"] = 0.10
    p90_params = dict(base_params)
    p90_params["objective"] = "reg:quantileerror"
    p90_params["quantile_alpha"] = 0.90

    model_p50 = train_xgb(
        fit_train,
        fit_eval,
        features,
        p50_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        progress_label="p50_train",
    )
    model_p10 = train_xgb(
        fit_train,
        fit_eval,
        features,
        p10_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        progress_label="p10_train",
    )
    model_p90 = train_xgb(
        fit_train,
        fit_eval,
        features,
        p90_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        progress_label="p90_train",
    )

    dtest = xgb.DMatrix(test[features], label=test[TARGET_COLUMN])
    p50 = model_p50.predict(dtest)
    p10 = model_p10.predict(dtest)
    p90 = model_p90.predict(dtest)

    dcalib = xgb.DMatrix(calib_df[features], label=calib_df[TARGET_COLUMN])
    p50_val = model_p50.predict(dcalib)
    p10_val = model_p10.predict(dcalib)
    p90_val = model_p90.predict(dcalib)
    lower_val_raw, upper_val_raw = build_interval(p10_val, p50_val, p90_val, scale=1.0)
    lower_test_raw, upper_test_raw = build_interval(p10, p50, p90, scale=1.0)

    calibration_meta: dict[str, float | int | dict[int, float] | str] = {
        "calibration_method": args.calibration_method
    }
    if args.calibration_method == "scale":
        scale = estimate_calibration_scale(
            y_true=calib_df[TARGET_COLUMN].to_numpy(),
            p10_raw=p10_val,
            p50=p50_val,
            p90_raw=p90_val,
            target_coverage=args.target_coverage,
        )
        lower_cal, upper_cal = build_interval(p10, p50, p90, scale=scale)
        calibration_meta["calibration_scale"] = float(scale)
    elif args.calibration_method == "conformal_additive":
        delta = estimate_conformal_delta(
            calib_df[TARGET_COLUMN].to_numpy(),
            lower_val_raw,
            upper_val_raw,
            target_coverage=args.target_coverage,
        )
        lower_cal = lower_test_raw - delta
        upper_cal = upper_test_raw + delta
        calibration_meta["calibration_delta"] = float(delta)
    else:
        val_calib = pd.DataFrame(
            {
                "month": calib_df["month"].astype(int).to_numpy(),
                "y_true": calib_df[TARGET_COLUMN].to_numpy(),
                "lower_raw": lower_val_raw,
                "upper_raw": upper_val_raw,
            }
        )
        month_to_delta, global_delta = estimate_monthly_conformal_deltas(
            val_calib,
            target_coverage=args.target_coverage,
            min_samples=args.monthly_calib_min_samples,
        )
        test_months = test["month"].astype(int).to_numpy()
        per_row_delta = np.array([month_to_delta.get(int(m), global_delta) for m in test_months], dtype=float)
        lower_cal = lower_test_raw - per_row_delta
        upper_cal = upper_test_raw + per_row_delta
        calibration_meta["calibration_delta_global"] = float(global_delta)
        calibration_meta["calibration_delta_monthly"] = month_to_delta
        calibration_meta["calibration_delta_monthly_count"] = int(len(month_to_delta))
        calibration_meta["monthly_calib_min_samples"] = int(args.monthly_calib_min_samples)

    out = test.copy()
    out["y_pred_p10_raw"] = p10
    out["y_pred_p50"] = p50
    out["y_pred_p90_raw"] = p90
    out["y_pred_p10"] = lower_cal
    out["y_pred_p90"] = upper_cal
    out["y_pred_p10_uncalibrated"] = lower_test_raw
    out["y_pred_p90_uncalibrated"] = upper_test_raw
    out["inside"] = (
        (out[TARGET_COLUMN] >= out["y_pred_p10"]) & (out[TARGET_COLUMN] <= out["y_pred_p90"])
    ).astype(int)
    out["inside_uncalibrated"] = (
        (out[TARGET_COLUMN] >= out["y_pred_p10_uncalibrated"])
        & (out[TARGET_COLUMN] <= out["y_pred_p90_uncalibrated"])
    ).astype(int)
    out["interval_width"] = out["y_pred_p90"] - out["y_pred_p10"]
    out["interval_width_uncalibrated"] = out["y_pred_p90_uncalibrated"] - out["y_pred_p10_uncalibrated"]

    summary_uncalibrated = compute_interval_metrics(
        y_true=out[TARGET_COLUMN].to_numpy(),
        lower=out["y_pred_p10_uncalibrated"].to_numpy(),
        p50=out["y_pred_p50"].to_numpy(),
        upper=out["y_pred_p90_uncalibrated"].to_numpy(),
        crossing_rate_raw=float((out["y_pred_p10_raw"] > out["y_pred_p90_raw"]).mean()),
    )
    summary = compute_interval_metrics(
        y_true=out[TARGET_COLUMN].to_numpy(),
        lower=out["y_pred_p10"].to_numpy(),
        p50=out["y_pred_p50"].to_numpy(),
        upper=out["y_pred_p90"].to_numpy(),
        crossing_rate_raw=float((out["y_pred_p10_raw"] > out["y_pred_p90_raw"]).mean()),
    )
    summary["target_coverage"] = float(args.target_coverage)
    summary["calibration_method"] = args.calibration_method
    summary["coverage_gap"] = float(summary["coverage_p10_p90"]) - float(args.target_coverage)  # type: ignore[arg-type]
    summary["coverage_uncalibrated"] = float(summary_uncalibrated["coverage_p10_p90"])  # type: ignore[index]
    summary["coverage_gain_after_calibration"] = float(summary["coverage_p10_p90"]) - float(summary_uncalibrated["coverage_p10_p90"])  # type: ignore[index]
    summary.update(calibration_meta)
    summary["calibration_split_meta"] = calibration_split_meta

    out.to_csv(outdir / "predictions_test_intervals.csv", index=False)
    tmp_raw = out.copy()
    tmp_raw["inside"] = out["inside_uncalibrated"]
    tmp_raw["interval_width"] = out["interval_width_uncalibrated"]
    save_group_interval_metrics(tmp_raw, "month", outdir / "coverage_by_month_test_raw.csv")
    save_group_interval_metrics(out, "month", outdir / "coverage_by_month_test.csv")
    if station_col in out.columns:
        save_group_interval_metrics(tmp_raw, station_col, outdir / "coverage_by_station_test_raw.csv")
        save_group_interval_metrics(out, station_col, outdir / "coverage_by_station_test.csv")
    save_json(outdir / "summary_metrics_uncalibrated.json", summary_uncalibrated)
    save_json(outdir / "summary_metrics.json", summary)
    save_json(outdir / "features_used.json", features)
    save_json(
        outdir / "params_used.json",
        {"base": base_params, "p10": p10_params, "p50": p50_params, "p90": p90_params},
    )
    save_json(outdir / "calibration_meta.json", calibration_meta)

    plot_interval_width_hist(out["interval_width"].to_numpy(), outdir / "interval_width_hist_test.png")
    plot_monthly_coverage(outdir / "coverage_by_month_test.csv", outdir / "coverage_by_month_test.png")

    print(f"Saved uncertainty run: {outdir}")


if __name__ == "__main__":
    main()
