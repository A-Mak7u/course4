from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from pipeline_common import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Улучшения bridge: gated, seasonal, nonlinear(XGB), downweight, uncertainty intervals"
    )
    parser.add_argument(
        "--input-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_vs_hydromet_overlap_2013_2023_allstations.csv",
    )
    parser.add_argument(
        "--selected-stations-file",
        default="transfer/hydromet_bridge_station_ids_selected.txt",
        help="Опциональный TXT со station id для фильтрации входного overlap (по одному id в строке).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--calib-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gate-eps", type=float, default=0.0, help="Минимальный выигрыш MAE на calib для открытия gate")
    parser.add_argument(
        "--heavy-threshold",
        type=float,
        default=0.03,
        help="Станция считается тяжёлой, если (bridge_mae - baseline_mae) на calib > threshold",
    )
    parser.add_argument(
        "--heavy-downweight",
        type=float,
        default=0.35,
        help="Вес train-сэмплов тяжёлых станций для ridge_downweight",
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-random-state", type=int, default=42)
    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["station"] = out["station"].astype(str).str.strip()
    out["T_rp5"] = pd.to_numeric(out["T_rp5"], errors="coerce")
    out["T_hydromet"] = pd.to_numeric(out["T_hydromet"], errors="coerce")
    out = out.dropna(subset=["Date", "station", "T_rp5", "T_hydromet"]).copy()

    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    out["rp5_x_sin"] = out["T_rp5"] * out["sin_doy"]
    out["rp5_x_cos"] = out["T_rp5"] * out["cos_doy"]
    out["is_cold"] = out["month"].isin([11, 12, 1, 2, 3]).astype(np.int8)
    return out.sort_values(["Date", "station"]).reset_index(drop=True)


def make_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_cols = ["T_rp5", "month", "sin_doy", "cos_doy", "rp5_x_sin", "rp5_x_cos"]
    dummies = pd.get_dummies(df["station"], prefix="station", drop_first=True, dtype=np.int8)
    design = pd.concat([df[base_cols], dummies], axis=1)
    return design, list(design.columns)


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    model = Ridge(alpha=alpha)
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_pred)


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    model = xgb.XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=args.xgb_random_state,
        n_jobs=8,
    )
    model.fit(X_train, y_train)
    return model.predict(X_pred)


def station_mae_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for station, g in df.groupby("station"):
        rows.append(
            {
                "station": str(station),
                "n": int(len(g)),
                "baseline_mae": float(mean_absolute_error(g["T_hydromet"], g["T_rp5"])),
                "model_mae": float(mean_absolute_error(g["T_hydromet"], g[pred_col])),
                "mae_gain": float(
                    mean_absolute_error(g["T_hydromet"], g["T_rp5"])
                    - mean_absolute_error(g["T_hydromet"], g[pred_col])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_mae").reset_index(drop=True)


def build_conformal_intervals(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_col: str,
    target_coverages: tuple[float, ...] = (0.80, 0.85, 0.90),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calib = calib_df.copy()
    test = test_df.copy()
    calib["abs_err"] = (calib[pred_col] - calib["T_hydromet"]).abs()
    test["abs_err"] = (test[pred_col] - test["T_hydromet"]).abs()

    global_rows: list[dict[str, float]] = []
    monthly_rows: list[dict[str, float | int]] = []

    for cov in target_coverages:
        q_global = float(calib["abs_err"].quantile(cov))
        lower = test[pred_col] - q_global
        upper = test[pred_col] + q_global
        coverage = float(((test["T_hydromet"] >= lower) & (test["T_hydromet"] <= upper)).mean())
        width = float((upper - lower).mean())
        global_rows.append(
            {
                "method": "global_quantile",
                "target_coverage": cov,
                "achieved_coverage": coverage,
                "coverage_gap": coverage - cov,
                "mean_width": width,
            }
        )

        # Monthly conformal: separate quantile by calendar month with fallback to global
        q_by_month = calib.groupby("month")["abs_err"].quantile(cov).to_dict()
        q_month = test["month"].map(lambda m: q_by_month.get(int(m), q_global)).astype(float)
        lower_m = test[pred_col] - q_month
        upper_m = test[pred_col] + q_month
        coverage_m = float(((test["T_hydromet"] >= lower_m) & (test["T_hydromet"] <= upper_m)).mean())
        width_m = float((upper_m - lower_m).mean())
        global_rows.append(
            {
                "method": "monthly_conformal",
                "target_coverage": cov,
                "achieved_coverage": coverage_m,
                "coverage_gap": coverage_m - cov,
                "mean_width": width_m,
            }
        )

        for month, g in test.groupby("month"):
            q_m = float(q_by_month.get(int(month), q_global))
            lo = g[pred_col] - q_m
            hi = g[pred_col] + q_m
            cov_m = float(((g["T_hydromet"] >= lo) & (g["T_hydromet"] <= hi)).mean())
            monthly_rows.append(
                {
                    "target_coverage": cov,
                    "month": int(month),
                    "n": int(len(g)),
                    "q_month": q_m,
                    "coverage": cov_m,
                    "mean_width": float((hi - lo).mean()),
                }
            )

    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows)


def plot_variant_compare(df_metrics: pd.DataFrame, outdir: Path) -> None:
    test = df_metrics[df_metrics["split"] == "test"].copy()
    order = test.sort_values("RMSE")["variant"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = test.set_index("variant").loc[order, "RMSE"]
    ax.bar(np.arange(len(order)), vals.values)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Bridge variants on test (2022-2023): RMSE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "variant_rmse_test.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = test.set_index("variant").loc[order, "MAE"]
    ax.bar(np.arange(len(order)), vals.values)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Bridge variants on test (2022-2023): MAE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "variant_mae_test.png", dpi=140)
    plt.close(fig)


def plot_interval_diagnostics(interval_global: pd.DataFrame, interval_monthly: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, g in interval_global.groupby("method"):
        gg = g.sort_values("target_coverage")
        ax.plot(gg["target_coverage"], gg["achieved_coverage"], marker="o", label=method)
    ax.plot([0.75, 0.95], [0.75, 0.95], linestyle="--", linewidth=1.1, label="ideal")
    ax.set_xlabel("target coverage")
    ax.set_ylabel("achieved coverage")
    ax.set_title("Uncertainty intervals: target vs achieved coverage")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "intervals_target_vs_achieved.png", dpi=140)
    plt.close(fig)

    focus_cov = 0.85
    g = interval_monthly[interval_monthly["target_coverage"] == focus_cov].sort_values("month")
    if len(g):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(g["month"], g["coverage"], marker="o")
        ax.axhline(focus_cov, linestyle="--", linewidth=1.1)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("month")
        ax.set_ylabel("coverage")
        ax.set_title("Monthly conformal coverage by month (target=0.85)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / "intervals_monthly_coverage_085.png", dpi=140)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    sel_path = Path(args.selected_stations_file) if args.selected_stations_file else None
    if sel_path and sel_path.exists():
        selected = [line.strip() for line in sel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if selected:
            selected_set = set(selected)
            df["station"] = df["station"].astype(str)
            df = df[df["station"].isin(selected_set)].copy()
    df = add_features(df)

    train_mask = df["year"] <= args.train_end_year
    calib_mask = df["year"] == args.calib_year
    test_mask = (df["year"] >= args.test_start_year) & (df["year"] <= args.test_end_year)

    train = df.loc[train_mask].copy()
    calib = df.loc[calib_mask].copy()
    test = df.loc[test_mask].copy()
    if train.empty or calib.empty or test.empty:
        raise RuntimeError("Одна из выборок train/calib/test пуста.")

    design_all, feature_cols = make_design(df)
    X_train = design_all.loc[train.index, feature_cols]
    X_calib = design_all.loc[calib.index, feature_cols]
    X_test = design_all.loc[test.index, feature_cols]
    y_train = train["T_hydromet"]
    y_calib = calib["T_hydromet"]
    y_test = test["T_hydromet"]

    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"outputs_runs/{ts}_rp5_hydromet_bridge_improvements_selected125")
    ensure_dir(outdir)

    pred_calib: dict[str, np.ndarray] = {}
    pred_test: dict[str, np.ndarray] = {}

    # 0) baseline
    pred_calib["baseline"] = calib["T_rp5"].to_numpy()
    pred_test["baseline"] = test["T_rp5"].to_numpy()

    # 1) ridge global
    pred_calib["ridge_global"] = fit_ridge(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_calib,
        alpha=args.ridge_alpha,
    )
    pred_test["ridge_global"] = fit_ridge(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_test,
        alpha=args.ridge_alpha,
    )

    # 2) gated ridge by station
    calib_eval = calib[["station", "T_hydromet", "T_rp5"]].copy()
    calib_eval["ridge_pred"] = pred_calib["ridge_global"]
    st_rows: list[dict[str, float | str]] = []
    for station, g in calib_eval.groupby("station"):
        b_mae = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        r_mae = float(mean_absolute_error(g["T_hydromet"], g["ridge_pred"]))
        st_rows.append({"station": station, "baseline_mae": b_mae, "ridge_mae": r_mae, "gain": b_mae - r_mae})
    st_df = pd.DataFrame(st_rows)
    gate_open = set(st_df.loc[st_df["gain"] > args.gate_eps, "station"].astype(str))

    pred_calib["ridge_gated"] = np.where(
        calib["station"].astype(str).isin(gate_open),
        pred_calib["ridge_global"],
        calib["T_rp5"].to_numpy(),
    )
    pred_test["ridge_gated"] = np.where(
        test["station"].astype(str).isin(gate_open),
        pred_test["ridge_global"],
        test["T_rp5"].to_numpy(),
    )

    # 3) seasonal ridge
    cold_train_mask = train["is_cold"] == 1
    warm_train_mask = train["is_cold"] == 0
    cold_model = Ridge(alpha=args.ridge_alpha).fit(X_train.loc[cold_train_mask], y_train.loc[cold_train_mask])
    warm_model = Ridge(alpha=args.ridge_alpha).fit(X_train.loc[warm_train_mask], y_train.loc[warm_train_mask])

    pred_calib_season = np.empty(len(calib), dtype=float)
    calib_cold = calib["is_cold"] == 1
    pred_calib_season[calib_cold.to_numpy()] = cold_model.predict(X_calib.loc[calib_cold])
    pred_calib_season[(~calib_cold).to_numpy()] = warm_model.predict(X_calib.loc[~calib_cold])
    pred_calib["ridge_seasonal"] = pred_calib_season

    pred_test_season = np.empty(len(test), dtype=float)
    test_cold = test["is_cold"] == 1
    pred_test_season[test_cold.to_numpy()] = cold_model.predict(X_test.loc[test_cold])
    pred_test_season[(~test_cold).to_numpy()] = warm_model.predict(X_test.loc[~test_cold])
    pred_test["ridge_seasonal"] = pred_test_season

    # 4) nonlinear xgb
    pred_calib["xgb_global"] = fit_xgb(X_train=X_train, y_train=y_train, X_pred=X_calib, args=args)
    pred_test["xgb_global"] = fit_xgb(X_train=X_train, y_train=y_train, X_pred=X_test, args=args)

    # 5) xgb gated by station
    calib_eval_x = calib[["station", "T_hydromet", "T_rp5"]].copy()
    calib_eval_x["xgb_pred"] = pred_calib["xgb_global"]
    st_rows_x: list[dict[str, float | str]] = []
    for station, g in calib_eval_x.groupby("station"):
        b_mae = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        x_mae = float(mean_absolute_error(g["T_hydromet"], g["xgb_pred"]))
        st_rows_x.append({"station": station, "baseline_mae": b_mae, "xgb_mae": x_mae, "gain": b_mae - x_mae})
    st_df_x = pd.DataFrame(st_rows_x)
    gate_open_x = set(st_df_x.loc[st_df_x["gain"] > args.gate_eps, "station"].astype(str))

    pred_calib["xgb_gated"] = np.where(
        calib["station"].astype(str).isin(gate_open_x),
        pred_calib["xgb_global"],
        calib["T_rp5"].to_numpy(),
    )
    pred_test["xgb_gated"] = np.where(
        test["station"].astype(str).isin(gate_open_x),
        pred_test["xgb_global"],
        test["T_rp5"].to_numpy(),
    )

    # 6) ridge downweight for heavy stations
    heavy_stations = set(
        st_df.loc[(st_df["ridge_mae"] - st_df["baseline_mae"]) > args.heavy_threshold, "station"].astype(str)
    )
    train_w = np.where(train["station"].astype(str).isin(heavy_stations), args.heavy_downweight, 1.0)
    pred_calib["ridge_downweight"] = fit_ridge(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_calib,
        alpha=args.ridge_alpha,
        sample_weight=train_w,
    )
    pred_test["ridge_downweight"] = fit_ridge(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_test,
        alpha=args.ridge_alpha,
        sample_weight=train_w,
    )

    # metrics table
    metric_rows: list[dict[str, float | int | str]] = []
    for variant in pred_test.keys():
        metric_rows.append({"variant": variant, "split": "calib", **compute_metrics(y_calib, pred_calib[variant])})
        metric_rows.append({"variant": variant, "split": "test", **compute_metrics(y_test, pred_test[variant])})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "variant_metrics.csv", index=False)

    # choose best by calib MAE among advanced variants (exclude baseline)
    calib_rank = metrics_df[(metrics_df["split"] == "calib") & (metrics_df["variant"] != "baseline")].sort_values("MAE")
    best_variant = str(calib_rank.iloc[0]["variant"])

    # station diagnostics for best variant
    test_diag = test[["Date", "station", "month", "T_hydromet", "T_rp5"]].copy()
    test_diag["pred_best"] = pred_test[best_variant]
    st_diag = station_mae_table(test_diag, pred_col="pred_best")
    st_diag.to_csv(outdir / "best_variant_station_metrics.csv", index=False)

    # monthly diagnostics for best variant
    monthly_rows: list[dict[str, float | int]] = []
    for month, g in test_diag.groupby("month"):
        baseline_mae = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        best_mae = float(mean_absolute_error(g["T_hydromet"], g["pred_best"]))
        monthly_rows.append(
            {
                "month": int(month),
                "n": int(len(g)),
                "baseline_mae": baseline_mae,
                "best_mae": best_mae,
                "mae_gain": baseline_mae - best_mae,
            }
        )
    monthly_df = pd.DataFrame(monthly_rows).sort_values("month")
    monthly_df.to_csv(outdir / "best_variant_monthly_metrics.csv", index=False)

    # uncertainty intervals for best variant (quantile + monthly conformal)
    calib_diag = calib[["Date", "station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_diag["pred_best"] = pred_calib[best_variant]
    interval_global, interval_monthly = build_conformal_intervals(calib_diag, test_diag, pred_col="pred_best")
    interval_global.to_csv(outdir / "intervals_summary.csv", index=False)
    interval_monthly.to_csv(outdir / "intervals_by_month.csv", index=False)

    # plots
    plot_variant_compare(metrics_df, outdir)
    plot_interval_diagnostics(interval_global, interval_monthly, outdir)

    # save gating / heavy lists
    pd.DataFrame({"station": sorted(gate_open)}).to_csv(outdir / "ridge_gated_open_stations.csv", index=False)
    pd.DataFrame({"station": sorted(gate_open_x)}).to_csv(outdir / "xgb_gated_open_stations.csv", index=False)
    pd.DataFrame({"station": sorted(heavy_stations)}).to_csv(outdir / "ridge_heavy_stations.csv", index=False)
    st_df.sort_values("gain", ascending=False).to_csv(outdir / "ridge_station_gain_on_calib.csv", index=False)
    st_df_x.sort_values("gain", ascending=False).to_csv(outdir / "xgb_station_gain_on_calib.csv", index=False)

    # summary json
    best_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == best_variant)].iloc[0].to_dict()
    base_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == "baseline")].iloc[0].to_dict()
    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "train_end_year": int(args.train_end_year),
        "calib_year": int(args.calib_year),
        "test_start_year": int(args.test_start_year),
        "test_end_year": int(args.test_end_year),
        "rows": {
            "train": int(len(train)),
            "calib": int(len(calib)),
            "test": int(len(test)),
            "stations_total": int(df["station"].nunique()),
        },
        "variants": metrics_df["variant"].drop_duplicates().tolist(),
        "best_variant_by_calib_mae": best_variant,
        "best_test_metrics": best_test,
        "baseline_test_metrics": base_test,
        "best_minus_baseline_test": {
            "RMSE_delta": float(best_test["RMSE"] - base_test["RMSE"]),
            "MAE_delta": float(best_test["MAE"] - base_test["MAE"]),
            "R2_delta": float(best_test["R2"] - base_test["R2"]),
        },
        "ridge_gated_open_station_count": int(len(gate_open)),
        "xgb_gated_open_station_count": int(len(gate_open_x)),
        "ridge_heavy_station_count": int(len(heavy_stations)),
    }
    save_json(outdir / "summary.json", summary)

    print(f"Saved improvement run: {outdir}")
    print(f"Best variant by calib MAE: {best_variant}")
    print(
        "Test delta vs baseline:",
        f"RMSE={summary['best_minus_baseline_test']['RMSE_delta']:.6f},",
        f"MAE={summary['best_minus_baseline_test']['MAE_delta']:.6f},",
        f"R2={summary['best_minus_baseline_test']['R2_delta']:.6f}",
    )


if __name__ == "__main__":
    main()
