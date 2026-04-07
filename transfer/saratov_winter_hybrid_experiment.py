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
    make_dmatrix,
    resolve_feature_list,
    save_json,
    split_by_year,
    train_xgb,
    tune_xgb,
)

WINTER_MONTHS = (11, 12, 1, 2, 3)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Saratov winter-hybrid experiment: full-year model + winter-only specialist"
    )
    parser.add_argument("--csv", default="final_2013_2023_T_ERA5_LST_daynight.csv")
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials-full", type=int, default=30)
    parser.add_argument("--n-trials-winter", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-inflated-precip", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def _month_mae(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    tmp = df[["month"]].copy()
    tmp["__y"] = y_true
    tmp["__p"] = y_pred
    out = (
        tmp.groupby("month", as_index=False)
        .apply(lambda s: pd.Series({"n": int(len(s)), "MAE": float(np.mean(np.abs(s["__p"] - s["__y"])))}))
        .reset_index(drop=True)
    )
    out["month"] = out["month"].astype(int)
    return out.sort_values("month")


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir or f"outputs_runs/{ts}_saratov_winter_hybrid")
    ensure_dir(outdir)

    df, meta = load_dataset(args.csv)
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df[meta.date_col]).dt.year

    train_mask_base, _ = split_by_year(
        df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    df = build_feature_frame(
        df,
        meta,
        train_mask=train_mask_base,
        include_station_mean=True,
        zero_inflated_precip=args.zero_inflated_precip,
    )
    features = resolve_feature_list(df, include_station_mean=True)

    train_mask, test_mask = split_by_year(
        df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )
    train_full = df.loc[train_mask].dropna(subset=[TARGET_COLUMN]).copy()
    test_full = df.loc[test_mask].dropna(subset=[TARGET_COLUMN]).copy()
    if train_full.empty or test_full.empty:
        raise RuntimeError("train/test пустые после split")

    val_year = choose_validation_year(train_full)
    inner_train_full = train_full[train_full["year"] < val_year].copy()
    inner_val_full = train_full[train_full["year"] == val_year].copy()
    if inner_train_full.empty or inner_val_full.empty:
        raise RuntimeError("inner_train/inner_val для full пусты")

    print(
        f"[winter-hybrid] full_tune start rows_train={len(inner_train_full)} rows_val={len(inner_val_full)} features={len(features)}",
        flush=True,
    )
    full_params = tune_xgb(
        inner_train_full,
        inner_val_full,
        features,
        device=args.device,
        n_trials=args.n_trials_full,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        progress_label="full_tune",
    )
    full_model = train_xgb(
        train_full,
        inner_val_full,
        features,
        full_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=100,
        progress_label="full_train",
    )

    train_winter = train_full[train_full["month"].isin(WINTER_MONTHS)].copy()
    test_winter = test_full[test_full["month"].isin(WINTER_MONTHS)].copy()
    inner_train_winter = train_winter[train_winter["year"] < val_year].copy()
    inner_val_winter = train_winter[train_winter["year"] == val_year].copy()
    if inner_train_winter.empty or inner_val_winter.empty or test_winter.empty:
        raise RuntimeError("winter split пустой, эксперимент невозможен")

    print(
        f"[winter-hybrid] winter_tune start rows_train={len(inner_train_winter)} rows_val={len(inner_val_winter)} features={len(features)}",
        flush=True,
    )
    winter_params = tune_xgb(
        inner_train_winter,
        inner_val_winter,
        features,
        device=args.device,
        n_trials=args.n_trials_winter,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        progress_label="winter_tune",
    )
    winter_model = train_xgb(
        train_winter,
        inner_val_winter,
        features,
        winter_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=100,
        progress_label="winter_train",
    )

    pred_full_test, metrics_full_test = evaluate_model(full_model, test_full, features)

    winter_mask = test_full["month"].isin(WINTER_MONTHS).to_numpy()
    pred_hybrid_test = pred_full_test.copy()
    pred_hybrid_test[winter_mask] = winter_model.predict(make_dmatrix(test_full.loc[winter_mask], features))

    y_test = test_full[TARGET_COLUMN].to_numpy()
    metrics_hybrid_test = compute_metrics(y_test, pred_hybrid_test)

    y_test_winter = y_test[winter_mask]
    pred_full_winter = pred_full_test[winter_mask]
    pred_hybrid_winter = pred_hybrid_test[winter_mask]
    metrics_full_winter = compute_metrics(y_test_winter, pred_full_winter)
    metrics_hybrid_winter = compute_metrics(y_test_winter, pred_hybrid_winter)

    nonwinter_mask = ~winter_mask
    y_test_nonwinter = y_test[nonwinter_mask]
    pred_full_nonwinter = pred_full_test[nonwinter_mask]
    pred_hybrid_nonwinter = pred_hybrid_test[nonwinter_mask]
    metrics_full_nonwinter = compute_metrics(y_test_nonwinter, pred_full_nonwinter)
    metrics_hybrid_nonwinter = compute_metrics(y_test_nonwinter, pred_hybrid_nonwinter)

    summary = {
        "baseline_full_test": metrics_full_test,
        "hybrid_full_test": metrics_hybrid_test,
        "baseline_winter_test": metrics_full_winter,
        "hybrid_winter_test": metrics_hybrid_winter,
        "baseline_nonwinter_test": metrics_full_nonwinter,
        "hybrid_nonwinter_test": metrics_hybrid_nonwinter,
        "delta_winter_mae": float(metrics_hybrid_winter["MAE"] - metrics_full_winter["MAE"]),
        "delta_winter_rmse": float(metrics_hybrid_winter["RMSE"] - metrics_full_winter["RMSE"]),
        "delta_full_mae": float(metrics_hybrid_test["MAE"] - metrics_full_test["MAE"]),
        "delta_full_rmse": float(metrics_hybrid_test["RMSE"] - metrics_full_test["RMSE"]),
    }

    summary_rows = [
        {"slice": "test_full", "model": "baseline_full", **metrics_full_test},
        {"slice": "test_full", "model": "hybrid_full+winter", **metrics_hybrid_test},
        {"slice": "test_winter", "model": "baseline_full", **metrics_full_winter},
        {"slice": "test_winter", "model": "hybrid_full+winter", **metrics_hybrid_winter},
        {"slice": "test_nonwinter", "model": "baseline_full", **metrics_full_nonwinter},
        {"slice": "test_nonwinter", "model": "hybrid_full+winter", **metrics_hybrid_nonwinter},
    ]
    pd.DataFrame(summary_rows).to_csv(outdir / "summary_metrics.csv", index=False)
    save_json(outdir / "summary_metrics.json", summary)
    save_json(outdir / "params_full.json", full_params)
    save_json(outdir / "params_winter.json", winter_params)
    save_json(outdir / "run_config.json", vars(args))
    save_json(outdir / "features_used.json", list(features))

    pred_df = test_full.copy()
    pred_df["y_true"] = y_test
    pred_df["pred_baseline_full"] = pred_full_test
    pred_df["pred_hybrid"] = pred_hybrid_test
    pred_df["is_winter"] = winter_mask.astype(np.int8)
    pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    baseline_by_month = _month_mae(test_full, y_test, pred_full_test).rename(columns={"MAE": "MAE_baseline"})
    hybrid_by_month = _month_mae(test_full, y_test, pred_hybrid_test).rename(columns={"MAE": "MAE_hybrid"})
    month_cmp = baseline_by_month.merge(hybrid_by_month[["month", "MAE_hybrid"]], on="month", how="inner")
    month_cmp["MAE_delta_hybrid_minus_baseline"] = month_cmp["MAE_hybrid"] - month_cmp["MAE_baseline"]
    month_cmp.to_csv(outdir / "mae_by_month_comparison.csv", index=False)

    plt.figure(figsize=(8.5, 4.5))
    plt.plot(month_cmp["month"], month_cmp["MAE_baseline"], marker="o", label="baseline")
    plt.plot(month_cmp["month"], month_cmp["MAE_hybrid"], marker="o", label="hybrid")
    plt.xlabel("Month")
    plt.ylabel("MAE (test)")
    plt.title("Month-wise MAE: baseline vs winter-hybrid")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mae_by_month_comparison.png", dpi=160)
    plt.close()

    print(f"Saved winter-hybrid run: {outdir}", flush=True)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
