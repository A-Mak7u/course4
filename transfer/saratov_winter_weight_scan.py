from __future__ import annotations

import argparse
import datetime as dt
import json
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
    compute_metrics,
    ensure_dir,
    load_dataset,
    resolve_feature_list,
    split_by_year,
)

WINTER_MONTHS = (11, 12, 1, 2, 3)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Saratov full-model winter weight scan (one model, weighted winter rows)"
    )
    parser.add_argument("--csv", default="final_2013_2023_T_ERA5_LST_daynight.csv")
    parser.add_argument("--params-json", default="outputs_runs/20260407_163025_saratov_winter_hybrid/params_full.json")
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--factors", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--zero-inflated-precip", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def _load_params(path: str | Path) -> dict[str, float | int | str]:
    with open(path, "r", encoding="utf-8") as handle:
        params = json.load(handle)
    return params


def _plot_metric_scan(summary: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(8.2, 4.5))
    plt.plot(summary["factor"], summary["RMSE_full"], marker="o", label="RMSE full")
    plt.plot(summary["factor"], summary["RMSE_winter"], marker="o", label="RMSE winter")
    plt.xlabel("Winter weight factor")
    plt.ylabel("RMSE (test)")
    plt.title("Winter weight scan")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir or f"outputs_runs/{ts}_saratov_winter_weight_scan")
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
        include_station_mean=False,
        zero_inflated_precip=args.zero_inflated_precip,
    )

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
        raise RuntimeError("train/test пустые")

    train = apply_station_train_mean_from_reference(train, train, meta)
    test = apply_station_train_mean_from_reference(test, train, meta)
    features = resolve_feature_list(train, include_station_mean=True)

    val_year = int(train["year"].max())
    inner_val = train[train["year"] == val_year].copy()
    if inner_val.empty:
        raise RuntimeError("inner_val пустой")

    params = _load_params(args.params_json)

    dval = xgb.DMatrix(inner_val[features], label=inner_val[TARGET_COLUMN])
    dtest = xgb.DMatrix(test[features], label=test[TARGET_COLUMN])
    winter_mask_test = test["month"].isin(WINTER_MONTHS).to_numpy()
    y_test = test[TARGET_COLUMN].to_numpy()
    y_test_winter = y_test[winter_mask_test]

    rows: list[dict[str, float | int]] = []
    for factor in args.factors:
        weights = np.ones(len(train), dtype=np.float32)
        winter_mask_train = train["month"].isin(WINTER_MONTHS).to_numpy()
        weights[winter_mask_train] = float(factor)

        dtrain = xgb.DMatrix(train[features], label=train[TARGET_COLUMN], weight=weights)
        print(
            f"[winter-weight-scan] factor={factor} train_rows={len(train)} winter_rows={int(winter_mask_train.sum())}",
            flush=True,
        )
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=200,
        )

        pred_test = model.predict(dtest)
        pred_test_winter = pred_test[winter_mask_test]

        m_full = compute_metrics(y_test, pred_test)
        m_winter = compute_metrics(y_test_winter, pred_test_winter)

        rows.append(
            {
                "factor": float(factor),
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "RMSE_full": float(m_full["RMSE"]),
                "MAE_full": float(m_full["MAE"]),
                "R2_full": float(m_full["R2"]),
                "RMSE_winter": float(m_winter["RMSE"]),
                "MAE_winter": float(m_winter["MAE"]),
                "R2_winter": float(m_winter["R2"]),
                "n_full": int(m_full["n"]),
                "n_winter": int(m_winter["n"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values("factor")
    summary.to_csv(outdir / "summary_scan.csv", index=False)
    _plot_metric_scan(summary, outdir / "rmse_scan.png")

    best_winter = summary.loc[summary["RMSE_winter"].idxmin()].to_dict()
    best_full = summary.loc[summary["RMSE_full"].idxmin()].to_dict()

    with open(outdir / "best_by_metric.json", "w", encoding="utf-8") as handle:
        json.dump({"best_winter_rmse": best_winter, "best_full_rmse": best_full}, handle, indent=2, ensure_ascii=False)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print(f"Saved winter weight scan: {outdir}", flush=True)
    print(f"best_winter_rmse={best_winter}", flush=True)
    print(f"best_full_rmse={best_full}", flush=True)


if __name__ == "__main__":
    main()
