from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from pipeline_common import ensure_dir, save_json


VARIANT_LABELS_RU = {
    "baseline": "Базовая линия (RP5)",
    "ridge_global": "Ridge global",
    "ridge_gated": "Ridge + gate",
    "ridge_gated_station_month": "Ridge + gate(station+month)",
    "ridge_soft_station": "Ridge soft(station)",
    "ridge_soft_station_month": "Ridge soft(station+month)",
    "ridge_seasonal": "Ridge seasonal",
    "ridge_downweight": "Ridge downweight",
    "xgb_global": "XGB global",
    "xgb_gated": "XGB + gate",
    "xgb_gated_station_month": "XGB + gate(station+month)",
    "xgb_soft_station": "XGB soft(station)",
    "xgb_soft_station_month": "XGB soft(station+month)",
    "xgb_delta_global": "XGB delta global",
    "xgb_delta_gated": "XGB delta + gate",
    "xgb_delta_gated_adaptive": "XGB delta + adaptive gate",
    "xgb_delta_gated_adaptive_safeguard": "XGB delta + adaptive gate + safeguard",
    "xgb_delta_gated_station_month": "XGB delta + gate(station+month)",
    "xgb_delta_clustered_v2": "XGB delta clustered v2",
    "xgb_delta_clustered_v2_gated": "XGB delta clustered v2 + gate",
    "xgb_delta_clustered_v2_gated_adaptive": "XGB delta clustered v2 + adaptive gate",
    "xgb_delta_clustered_v3": "XGB delta clustered v3",
    "xgb_delta_clustered_v3_gated": "XGB delta clustered v3 + gate",
    "xgb_delta_clustered_v3_gated_adaptive": "XGB delta clustered v3 + adaptive gate",
}


def variant_label_ru(name: str) -> str:
    return VARIANT_LABELS_RU.get(str(name), str(name))


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
    parser.add_argument("--gate-eps", type=float, default=0.005, help="Минимальный выигрыш MAE на calib для открытия gate")
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
    parser.add_argument(
        "--min-station-month-samples",
        type=int,
        default=15,
        help="Минимум наблюдений в (station,month) на calib для station-month gate/blend.",
    )
    parser.add_argument(
        "--soft-scale-grid",
        default="0.5,1.0,1.5,2.0,3.0,4.0",
        help="Сетка scale для alpha в soft-blend (подбор на calib).",
    )
    parser.add_argument(
        "--conformal-station-groups",
        type=int,
        default=4,
        help="Число station-group для conditional conformal.",
    )
    parser.add_argument(
        "--conformal-min-group-month-samples",
        type=int,
        default=10,
        help="Минимум наблюдений в (station_group, month) на calib для conditional quantile.",
    )
    parser.add_argument(
        "--cluster-bridge-groups",
        type=int,
        default=4,
        help="Число station-кластеров для cluster bridge (дельта-модель).",
    )
    parser.add_argument(
        "--cluster-bridge-min-train-rows",
        type=int,
        default=1200,
        help="Минимум train-строк в кластере для обучения отдельной cluster-модели.",
    )
    parser.add_argument(
        "--cluster-v3-groups",
        type=int,
        default=5,
        help="Число station-кластеров для cluster bridge v3 (seasonal+bias+yearly-error).",
    )
    parser.add_argument(
        "--cluster-v3-min-train-rows",
        type=int,
        default=1200,
        help="Минимум train-строк в кластере для обучения отдельной cluster-v3 модели.",
    )
    parser.add_argument(
        "--adaptive-gate-enabled",
        action="store_true",
        help="Включить адаптивный station-wise gate для delta-модели по station-risk (LOSO/rolling).",
    )
    parser.add_argument(
        "--adaptive-gate-loso-csv",
        default=None,
        help="CSV LOSO-диагностики (ожидается station + loso_MAE_gain_vs_baseline).",
    )
    parser.add_argument(
        "--adaptive-gate-rolling-csv",
        default=None,
        help="CSV rolling-origin (ожидается xgb_delta_gated_MAE_gain_vs_baseline).",
    )
    parser.add_argument(
        "--adaptive-gate-loso-scale",
        type=float,
        default=0.20,
        help="Коэффициент штрафа к eps по station-risk из LOSO (для отрицательных gains).",
    )
    parser.add_argument(
        "--adaptive-gate-max-penalty",
        type=float,
        default=0.02,
        help="Максимальный station-wise штраф к eps в adaptive gate.",
    )
    parser.add_argument(
        "--adaptive-gate-rolling-scale",
        type=float,
        default=0.40,
        help="Коэффициент глобального штрафа к eps из волатильности rolling-origin.",
    )
    parser.add_argument("--adaptive-gate-min-eps", type=float, default=0.0)
    parser.add_argument("--adaptive-gate-max-eps", type=float, default=0.05)
    parser.add_argument(
        "--safeguard-margin",
        type=float,
        default=0.002,
        help="Минимальный запас MAE на calib, иначе включается fallback на baseline/seasonal.",
    )
    parser.add_argument("--run-rolling-origin", action="store_true", help="Считать rolling-origin диагностику.")
    parser.add_argument("--rolling-calib-start-year", type=int, default=2019)
    parser.add_argument("--rolling-calib-end-year", type=int, default=2022)
    parser.add_argument("--run-loso", action="store_true", help="Считать LOSO диагностику по станциям.")
    parser.add_argument(
        "--loso-max-stations",
        type=int,
        default=None,
        help="Ограничение числа станций для LOSO (по убыванию тестовых строк).",
    )
    parser.add_argument(
        "--diag-xgb-n-estimators",
        type=int,
        default=220,
        help="Число деревьев XGB для rolling/LOSO-диагностики (ускоренный режим).",
    )
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


def make_xgb_model(
    args: argparse.Namespace,
    n_estimators_override: int | None = None,
) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=int(n_estimators_override) if n_estimators_override is not None else args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=args.xgb_random_state,
        n_jobs=8,
    )


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    args: argparse.Namespace,
    n_estimators_override: int | None = None,
) -> np.ndarray:
    model = make_xgb_model(args=args, n_estimators_override=n_estimators_override)
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


def alpha_from_mae(baseline_mae: float, model_mae: float) -> float:
    if not np.isfinite(baseline_mae) or baseline_mae <= 1e-12:
        return 0.0
    gain = baseline_mae - model_mae
    if not np.isfinite(gain):
        return 0.0
    return float(np.clip(gain / baseline_mae, 0.0, 1.0))


def build_station_gain_table(calib_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for station, g in calib_df.groupby("station"):
        baseline_mae = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        model_mae = float(mean_absolute_error(g["T_hydromet"], g[pred_col]))
        rows.append(
            {
                "station": str(station),
                "n": int(len(g)),
                "baseline_mae": baseline_mae,
                "model_mae": model_mae,
                "gain": baseline_mae - model_mae,
                "alpha": alpha_from_mae(baseline_mae, model_mae),
            }
        )
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def build_station_month_gain_table(
    calib_df: pd.DataFrame,
    pred_col: str,
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (station, month), g in calib_df.groupby(["station", "month"]):
        baseline_mae = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        model_mae = float(mean_absolute_error(g["T_hydromet"], g[pred_col]))
        rows.append(
            {
                "station": str(station),
                "month": int(month),
                "n": int(len(g)),
                "baseline_mae": baseline_mae,
                "model_mae": model_mae,
                "gain": baseline_mae - model_mae,
                "alpha": alpha_from_mae(baseline_mae, model_mae),
                "is_eligible": int(len(g) >= min_samples),
            }
        )
    out = pd.DataFrame(rows).sort_values(["station", "month"]).reset_index(drop=True)
    return out


def build_alpha_vector(
    station_series: pd.Series,
    month_series: pd.Series,
    station_alpha: dict[str, float],
    station_month_alpha: dict[tuple[str, int], float] | None = None,
    default_alpha: float = 0.0,
) -> np.ndarray:
    station_vals = station_series.astype(str).tolist()
    month_vals = month_series.astype(int).tolist()
    out = np.empty(len(station_vals), dtype=float)
    for i, (s, m) in enumerate(zip(station_vals, month_vals)):
        if station_month_alpha is not None:
            a = station_month_alpha.get((s, int(m)))
            if a is not None:
                out[i] = a
                continue
        out[i] = station_alpha.get(s, default_alpha)
    return np.clip(out, 0.0, 1.0)


def parse_soft_scale_grid(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(float(token))
    vals = [v for v in vals if v > 0]
    if not vals:
        vals = [1.0]
    return sorted(set(vals))


def scale_alpha_map(alpha_map: dict, scale: float) -> dict:
    return {k: float(np.clip(v * scale, 0.0, 1.0)) for k, v in alpha_map.items()}


def tune_soft_scale_on_calib(
    y_true: pd.Series,
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    station_series: pd.Series,
    month_series: pd.Series,
    station_alpha: dict[str, float],
    station_month_alpha: dict[tuple[str, int], float] | None,
    scale_grid: list[float],
) -> tuple[float, np.ndarray]:
    best_scale = scale_grid[0]
    best_pred = apply_soft_blend(
        baseline_pred=baseline_pred,
        model_pred=model_pred,
        station_series=station_series,
        month_series=month_series,
        station_alpha=scale_alpha_map(station_alpha, best_scale),
        station_month_alpha=scale_alpha_map(station_month_alpha, best_scale) if station_month_alpha is not None else None,
        default_alpha=0.0,
    )
    best_mae = float(mean_absolute_error(y_true, best_pred))

    for scale in scale_grid[1:]:
        pred = apply_soft_blend(
            baseline_pred=baseline_pred,
            model_pred=model_pred,
            station_series=station_series,
            month_series=month_series,
            station_alpha=scale_alpha_map(station_alpha, scale),
            station_month_alpha=scale_alpha_map(station_month_alpha, scale) if station_month_alpha is not None else None,
            default_alpha=0.0,
        )
        mae = float(mean_absolute_error(y_true, pred))
        if mae < best_mae:
            best_mae = mae
            best_scale = scale
            best_pred = pred

    return float(best_scale), best_pred


def apply_soft_blend(
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    station_series: pd.Series,
    month_series: pd.Series,
    station_alpha: dict[str, float],
    station_month_alpha: dict[tuple[str, int], float] | None = None,
    default_alpha: float = 0.0,
) -> np.ndarray:
    alpha = build_alpha_vector(
        station_series=station_series,
        month_series=month_series,
        station_alpha=station_alpha,
        station_month_alpha=station_month_alpha,
        default_alpha=default_alpha,
    )
    return alpha * model_pred + (1.0 - alpha) * baseline_pred


def apply_hard_gate_station_month(
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    station_series: pd.Series,
    month_series: pd.Series,
    open_pairs: set[tuple[str, int]],
) -> np.ndarray:
    stations = station_series.astype(str).tolist()
    months = month_series.astype(int).tolist()
    mask = np.array([(s, int(m)) in open_pairs for s, m in zip(stations, months)], dtype=bool)
    out = baseline_pred.copy()
    out[mask] = model_pred[mask]
    return out


def _read_optional_csv(path: str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _rolling_volatility_penalty(rolling_df: pd.DataFrame, scale: float) -> float:
    if rolling_df.empty:
        return 0.0
    col = "xgb_delta_gated_MAE_gain_vs_baseline"
    if col not in rolling_df.columns:
        return 0.0
    vals = pd.to_numeric(rolling_df[col], errors="coerce").dropna()
    if vals.empty:
        return 0.0
    vol = float(vals.std(ddof=0))
    if not np.isfinite(vol):
        return 0.0
    return max(0.0, float(scale) * vol)


def build_adaptive_gate_table(
    station_gain_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float | int | bool]]:
    out = station_gain_df.copy()
    out["station"] = out["station"].astype(str)
    out["base_eps"] = float(args.gate_eps)
    out["loso_penalty"] = 0.0
    out["rolling_penalty"] = 0.0
    out["adaptive_eps"] = float(args.gate_eps)

    loso_df = _read_optional_csv(args.adaptive_gate_loso_csv)
    rolling_df = _read_optional_csv(args.adaptive_gate_rolling_csv)
    rolling_penalty = _rolling_volatility_penalty(rolling_df, args.adaptive_gate_rolling_scale)

    if args.adaptive_gate_enabled and (not loso_df.empty) and {"station", "loso_MAE_gain_vs_baseline"}.issubset(loso_df.columns):
        risk_map = (
            loso_df[["station", "loso_MAE_gain_vs_baseline"]]
            .dropna()
            .assign(station=lambda d: d["station"].astype(str))
            .drop_duplicates(subset=["station"], keep="last")
            .set_index("station")["loso_MAE_gain_vs_baseline"]
            .to_dict()
        )

        penalties: list[float] = []
        for station in out["station"].tolist():
            gain = risk_map.get(str(station), np.nan)
            if not np.isfinite(gain):
                penalties.append(0.0)
                continue
            pen = float(np.clip(max(0.0, -float(gain)) * args.adaptive_gate_loso_scale, 0.0, args.adaptive_gate_max_penalty))
            penalties.append(pen)
        out["loso_penalty"] = penalties

    if args.adaptive_gate_enabled:
        out["rolling_penalty"] = float(rolling_penalty)
        out["adaptive_eps"] = (
            out["base_eps"].astype(float)
            + out["loso_penalty"].astype(float)
            + out["rolling_penalty"].astype(float)
        ).clip(lower=float(args.adaptive_gate_min_eps), upper=float(args.adaptive_gate_max_eps))

    out["gate_open_constant"] = (out["gain"] > out["base_eps"]).astype(int)
    out["gate_open_adaptive"] = (out["gain"] > out["adaptive_eps"]).astype(int)

    eps_map = {str(r["station"]): float(r["adaptive_eps"]) for _, r in out.iterrows()}
    summary = {
        "adaptive_gate_enabled": bool(args.adaptive_gate_enabled),
        "base_eps": float(args.gate_eps),
        "adaptive_eps_mean": float(out["adaptive_eps"].mean()),
        "adaptive_eps_median": float(out["adaptive_eps"].median()),
        "rolling_penalty": float(rolling_penalty),
        "gate_open_constant_count": int(out["gate_open_constant"].sum()),
        "gate_open_adaptive_count": int(out["gate_open_adaptive"].sum()),
    }
    return out.sort_values("gain", ascending=False).reset_index(drop=True), eps_map, summary


def apply_station_gate_with_eps(
    station_series: pd.Series,
    gain_table: pd.DataFrame,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    eps_col: str,
) -> tuple[np.ndarray, set[str]]:
    gain_map = gain_table.set_index("station")["gain"].to_dict()
    eps_map = gain_table.set_index("station")[eps_col].to_dict()
    station_vals = station_series.astype(str).tolist()
    mask = np.array(
        [float(gain_map.get(st, -1e9)) > float(eps_map.get(st, 1e9)) for st in station_vals],
        dtype=bool,
    )
    out = baseline_pred.copy()
    out[mask] = model_pred[mask]
    open_st = {st for st in station_vals if float(gain_map.get(st, -1e9)) > float(eps_map.get(st, 1e9))}
    return out, open_st


def build_safeguard_station_policy(
    calib_df: pd.DataFrame,
    pred_col_model: str,
    pred_col_seasonal: str,
    margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for station, g in calib_df.groupby("station"):
        y = g["T_hydromet"]
        mae_model = float(mean_absolute_error(y, g[pred_col_model]))
        mae_baseline = float(mean_absolute_error(y, g["T_rp5"]))
        mae_seasonal = float(mean_absolute_error(y, g[pred_col_seasonal]))
        best_fallback = "baseline" if mae_baseline <= mae_seasonal else "seasonal"
        fallback_mae = min(mae_baseline, mae_seasonal)
        use_fallback = bool(mae_model > (fallback_mae - float(margin)))
        rows.append(
            {
                "station": str(station),
                "n_calib": int(len(g)),
                "mae_model": mae_model,
                "mae_baseline": mae_baseline,
                "mae_seasonal": mae_seasonal,
                "best_fallback": best_fallback,
                "fallback_mae": fallback_mae,
                "use_fallback": int(use_fallback),
                "calib_margin_to_fallback": float(mae_model - fallback_mae),
            }
        )
    return pd.DataFrame(rows).sort_values("calib_margin_to_fallback", ascending=False).reset_index(drop=True)


def apply_safeguard_policy(
    station_series: pd.Series,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    seasonal_pred: np.ndarray,
    policy_df: pd.DataFrame,
) -> np.ndarray:
    policy = policy_df.set_index("station")
    out = model_pred.copy()
    station_vals = station_series.astype(str).tolist()
    for i, st in enumerate(station_vals):
        if st not in policy.index:
            continue
        row = policy.loc[st]
        if int(row["use_fallback"]) != 1:
            continue
        if str(row["best_fallback"]) == "seasonal":
            out[i] = seasonal_pred[i]
        else:
            out[i] = baseline_pred[i]
    return out


def _baseline_error_profile(train_df: pd.DataFrame) -> pd.DataFrame:
    work = train_df.copy()
    work["err_base"] = work["T_rp5"] - work["T_hydromet"]
    work["abs_err_base"] = work["err_base"].abs()
    station_year = (
        work.groupby(["station", "year"], as_index=False)
        .agg(
            mae_year=("abs_err_base", "mean"),
            bias_year=("err_base", "mean"),
            n_year=("err_base", "size"),
        )
    )
    prof = (
        station_year.groupby("station", as_index=False)
        .agg(
            mae_year_mean=("mae_year", "mean"),
            mae_year_std=("mae_year", "std"),
            mae_year_max=("mae_year", "max"),
            bias_year_std=("bias_year", "std"),
            years_count=("year", "nunique"),
        )
    )
    for c in ["mae_year_std", "bias_year_std"]:
        prof[c] = prof[c].fillna(0.0)
    return prof


def build_station_cluster_map_v3(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    pred_calib_delta_global: np.ndarray,
    n_groups: int,
    random_state: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    calib_work = calib_df.copy()
    calib_work["pred_delta"] = pred_calib_delta_global
    calib_work["err_model"] = calib_work["pred_delta"] - calib_work["T_hydromet"]
    calib_work["err_base"] = calib_work["T_rp5"] - calib_work["T_hydromet"]
    calib_work["abs_err_model"] = calib_work["err_model"].abs()
    calib_work["abs_err_base"] = calib_work["err_base"].abs()
    calib_work["gain"] = calib_work["abs_err_base"] - calib_work["abs_err_model"]

    train_work = train_df.copy()
    train_work["err_base"] = train_work["T_rp5"] - train_work["T_hydromet"]
    train_work["abs_err_base"] = train_work["err_base"].abs()

    base_station = (
        train_work.groupby("station", as_index=False)
        .agg(
            rp5_mean=("T_rp5", "mean"),
            rp5_std=("T_rp5", "std"),
            hydromet_mean=("T_hydromet", "mean"),
            baseline_mae_train=("abs_err_base", "mean"),
            baseline_bias_train=("err_base", "mean"),
            n_train=("station", "size"),
        )
    )
    base_station["rp5_std"] = base_station["rp5_std"].fillna(0.0)

    cold_months = {11, 12, 1, 2, 3}
    train_work["is_cold"] = train_work["month"].isin(cold_months).astype(int)
    calib_work["is_cold"] = calib_work["month"].isin(cold_months).astype(int)

    train_mode = (
        train_work.groupby(["station", "is_cold"], as_index=False)
        .agg(mae_base_mode=("abs_err_base", "mean"), bias_base_mode=("err_base", "mean"))
    )
    train_mode = train_mode.pivot(index="station", columns="is_cold", values=["mae_base_mode", "bias_base_mode"])
    train_mode.columns = [f"{a}_{'cold' if b == 1 else 'warm'}" for a, b in train_mode.columns]
    train_mode = train_mode.reset_index()

    calib_station = (
        calib_work.groupby("station", as_index=False)
        .agg(
            calib_mae_model=("abs_err_model", "mean"),
            calib_mae_base=("abs_err_base", "mean"),
            calib_bias_model=("err_model", "mean"),
            calib_gain_mean=("gain", "mean"),
            n_calib=("station", "size"),
        )
    )
    calib_mode = (
        calib_work.groupby(["station", "is_cold"], as_index=False)
        .agg(
            calib_gain_mode=("gain", "mean"),
            calib_mae_model_mode=("abs_err_model", "mean"),
            calib_mae_base_mode=("abs_err_base", "mean"),
        )
    )
    calib_mode = calib_mode.pivot(index="station", columns="is_cold", values=["calib_gain_mode", "calib_mae_model_mode", "calib_mae_base_mode"])
    calib_mode.columns = [f"{a}_{'cold' if b == 1 else 'warm'}" for a, b in calib_mode.columns]
    calib_mode = calib_mode.reset_index()

    yearly = _baseline_error_profile(train_work)

    stats = base_station.merge(train_mode, on="station", how="left")
    stats = stats.merge(calib_station, on="station", how="left")
    stats = stats.merge(calib_mode, on="station", how="left")
    stats = stats.merge(yearly, on="station", how="left")
    for col in stats.columns:
        if col == "station":
            continue
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.0)

    feat_cols = [c for c in stats.columns if c != "station"]
    X = stats[feat_cols].copy()
    for col in feat_cols:
        mu = float(X[col].mean())
        sd = float(X[col].std())
        if not np.isfinite(sd) or sd <= 1e-12:
            X[col] = 0.0
        else:
            X[col] = (X[col] - mu) / sd

    n_eff = max(1, min(int(n_groups), len(stats)))
    if n_eff == 1:
        stats["cluster_id_v3"] = 0
    else:
        km = KMeans(n_clusters=n_eff, random_state=random_state, n_init=20)
        stats["cluster_id_v3"] = km.fit_predict(X.to_numpy())
    stats["cluster_id_v3"] = stats["cluster_id_v3"].astype(int)
    station_to_cluster = {str(r["station"]): int(r["cluster_id_v3"]) for _, r in stats.iterrows()}
    return station_to_cluster, stats


def build_station_group_map(
    calib_df: pd.DataFrame,
    pred_col: str,
    n_groups: int,
) -> dict[str, int]:
    work = calib_df.copy()
    work["abs_err"] = (work[pred_col] - work["T_hydromet"]).abs()
    station_order = (
        work.groupby("station")["abs_err"]
        .mean()
        .sort_values(ascending=True)
        .index.astype(str)
        .tolist()
    )
    if not station_order:
        return {}
    n_eff = max(1, min(int(n_groups), len(station_order)))
    chunks = np.array_split(np.array(station_order, dtype=object), n_eff)
    station_to_group: dict[str, int] = {}
    for gi, chunk in enumerate(chunks):
        for st in chunk.tolist():
            station_to_group[str(st)] = int(gi)
    return station_to_group


def build_station_cluster_map(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame | None,
    n_groups: int,
    random_state: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    work_train = train_df.copy()
    work_calib = calib_df.copy() if calib_df is not None else pd.DataFrame(columns=work_train.columns)
    work = pd.concat([work_train.assign(_split="train"), work_calib.assign(_split="calib")], ignore_index=True)
    if work.empty:
        return {}, pd.DataFrame()

    work["is_cold"] = work["month"].isin([11, 12, 1, 2, 3]).astype(int)
    grp = work.groupby("station")
    stats = pd.DataFrame({"station": grp.size().index.astype(str)})
    stats["n_rows"] = grp.size().values
    stats["n_train"] = work_train.groupby("station").size().reindex(stats["station"]).fillna(0).astype(int).to_numpy()
    stats["n_calib"] = work_calib.groupby("station").size().reindex(stats["station"]).fillna(0).astype(int).to_numpy()
    stats["rp5_mean"] = grp["T_rp5"].mean().values
    stats["rp5_std"] = grp["T_rp5"].std().fillna(0.0).values
    stats["hydromet_mean"] = grp["T_hydromet"].mean().values
    stats["baseline_bias"] = grp.apply(lambda g: (g["T_rp5"] - g["T_hydromet"]).mean()).values
    stats["baseline_mae"] = grp.apply(lambda g: mean_absolute_error(g["T_hydromet"], g["T_rp5"])).values

    def _safe_stat(g: pd.DataFrame, col: str) -> float:
        if g.empty:
            return 0.0
        return float(g[col].mean())

    cold_bias = work[work["is_cold"] == 1].groupby("station").apply(lambda g: _safe_stat(g, "T_rp5") - _safe_stat(g, "T_hydromet"))
    warm_bias = work[work["is_cold"] == 0].groupby("station").apply(lambda g: _safe_stat(g, "T_rp5") - _safe_stat(g, "T_hydromet"))
    cold_mae = work[work["is_cold"] == 1].groupby("station").apply(
        lambda g: float(mean_absolute_error(g["T_hydromet"], g["T_rp5"])) if len(g) else 0.0
    )
    warm_mae = work[work["is_cold"] == 0].groupby("station").apply(
        lambda g: float(mean_absolute_error(g["T_hydromet"], g["T_rp5"])) if len(g) else 0.0
    )
    stats["bias_cold"] = stats["station"].map(cold_bias).fillna(0.0).astype(float)
    stats["bias_warm"] = stats["station"].map(warm_bias).fillna(0.0).astype(float)
    stats["mae_cold"] = stats["station"].map(cold_mae).fillna(0.0).astype(float)
    stats["mae_warm"] = stats["station"].map(warm_mae).fillna(0.0).astype(float)
    stats["bias_seasonal_gap"] = stats["bias_cold"] - stats["bias_warm"]
    stats["mae_seasonal_gap"] = stats["mae_cold"] - stats["mae_warm"]

    month_pivot = work.pivot_table(index="station", columns="month", values="T_rp5", aggfunc="mean")
    winter_mean = month_pivot[[11, 12, 1, 2, 3]].mean(axis=1, skipna=True) if set([11, 12, 1, 2, 3]).intersection(month_pivot.columns) else pd.Series(dtype=float)
    summer_mean = month_pivot[[6, 7, 8]].mean(axis=1, skipna=True) if set([6, 7, 8]).intersection(month_pivot.columns) else pd.Series(dtype=float)
    stats["rp5_summer_minus_winter"] = (
        stats["station"].map(summer_mean).fillna(0.0).astype(float) - stats["station"].map(winter_mean).fillna(0.0).astype(float)
    )

    if stats.empty:
        return {}, stats

    feat_cols = [
        "rp5_mean",
        "rp5_std",
        "hydromet_mean",
        "baseline_mae",
        "baseline_bias",
        "bias_cold",
        "bias_warm",
        "mae_cold",
        "mae_warm",
        "bias_seasonal_gap",
        "mae_seasonal_gap",
        "rp5_summer_minus_winter",
    ]
    X = stats[feat_cols].copy()
    for col in feat_cols:
        mu = float(X[col].mean())
        sd = float(X[col].std())
        if not np.isfinite(sd) or sd <= 1e-12:
            X[col] = 0.0
        else:
            X[col] = (X[col] - mu) / sd

    n_eff = max(1, min(int(n_groups), len(stats)))
    if n_eff == 1:
        stats["cluster_id"] = 0
    else:
        km = KMeans(n_clusters=n_eff, random_state=random_state, n_init=20)
        stats["cluster_id"] = km.fit_predict(X.to_numpy())
    stats["cluster_id"] = stats["cluster_id"].astype(int)
    station_to_cluster = {str(r["station"]): int(r["cluster_id"]) for _, r in stats.iterrows()}
    return station_to_cluster, stats


def interval_quality_stats(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> tuple[float, float]:
    alpha = max(1e-6, 1.0 - float(target_coverage))
    y = y_true.astype(float)
    lo = lower.astype(float)
    hi = upper.astype(float)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    wis = width + (2.0 / alpha) * (below + above)
    crps_like = wis * (alpha / 2.0)
    return float(np.mean(wis)), float(np.mean(crps_like))


def build_conformal_intervals(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_col: str,
    target_coverages: tuple[float, ...] = (0.80, 0.85, 0.90),
    station_groups: int = 3,
    min_group_month_samples: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calib = calib_df.copy()
    test = test_df.copy()
    calib["abs_err"] = (calib[pred_col] - calib["T_hydromet"]).abs()
    test["abs_err"] = (test[pred_col] - test["T_hydromet"]).abs()

    station_group_map = build_station_group_map(
        calib_df=calib,
        pred_col=pred_col,
        n_groups=station_groups,
    )
    calib["station_group"] = calib["station"].astype(str).map(station_group_map).fillna(-1).astype(int)
    test["station_group"] = test["station"].astype(str).map(station_group_map).fillna(-1).astype(int)

    global_rows: list[dict[str, float]] = []
    monthly_rows: list[dict[str, float | int]] = []
    conditional_rows: list[dict[str, float | int | str]] = []

    for cov in target_coverages:
        q_global = float(calib["abs_err"].quantile(cov))
        lower = test[pred_col] - q_global
        upper = test[pred_col] + q_global
        coverage = float(((test["T_hydromet"] >= lower) & (test["T_hydromet"] <= upper)).mean())
        width = float((upper - lower).mean())
        wis_mean, crps_like = interval_quality_stats(
            y_true=test["T_hydromet"].to_numpy(),
            lower=lower.to_numpy(),
            upper=upper.to_numpy(),
            target_coverage=float(cov),
        )
        global_rows.append(
            {
                "method": "global_quantile",
                "target_coverage": cov,
                "achieved_coverage": coverage,
                "coverage_gap": coverage - cov,
                "mean_width": width,
                "wis_mean": wis_mean,
                "crps_like": crps_like,
            }
        )

        # Monthly conformal: separate quantile by calendar month with fallback to global
        q_by_month = calib.groupby("month")["abs_err"].quantile(cov).to_dict()
        q_month = test["month"].map(lambda m: q_by_month.get(int(m), q_global)).astype(float)
        lower_m = test[pred_col] - q_month
        upper_m = test[pred_col] + q_month
        coverage_m = float(((test["T_hydromet"] >= lower_m) & (test["T_hydromet"] <= upper_m)).mean())
        width_m = float((upper_m - lower_m).mean())
        wis_monthly, crps_monthly = interval_quality_stats(
            y_true=test["T_hydromet"].to_numpy(),
            lower=lower_m.to_numpy(),
            upper=upper_m.to_numpy(),
            target_coverage=float(cov),
        )
        global_rows.append(
            {
                "method": "monthly_conformal",
                "target_coverage": cov,
                "achieved_coverage": coverage_m,
                "coverage_gap": coverage_m - cov,
                "mean_width": width_m,
                "wis_mean": wis_monthly,
                "crps_like": crps_monthly,
            }
        )

        for month, g in test.groupby("month"):
            q_m = float(q_by_month.get(int(month), q_global))
            lo = g[pred_col] - q_m
            hi = g[pred_col] + q_m
            cov_m = float(((g["T_hydromet"] >= lo) & (g["T_hydromet"] <= hi)).mean())
            wis_m, crps_m = interval_quality_stats(
                y_true=g["T_hydromet"].to_numpy(),
                lower=lo.to_numpy(),
                upper=hi.to_numpy(),
                target_coverage=float(cov),
            )
            monthly_rows.append(
                {
                    "target_coverage": cov,
                    "month": int(month),
                    "n": int(len(g)),
                    "q_month": q_m,
                    "coverage": cov_m,
                    "mean_width": float((hi - lo).mean()),
                    "wis_mean": wis_m,
                    "crps_like": crps_m,
                }
            )

        # Conditional conformal: station_group + month with fallbacks
        q_group_month = calib.groupby(["station_group", "month"])["abs_err"].quantile(cov).to_dict()
        n_group_month = calib.groupby(["station_group", "month"]).size().to_dict()
        q_group = calib.groupby("station_group")["abs_err"].quantile(cov).to_dict()
        n_group = calib.groupby("station_group").size().to_dict()
        q_month = calib.groupby("month")["abs_err"].quantile(cov).to_dict()

        cond_q = np.empty(len(test), dtype=float)
        cond_src: list[str] = []
        for i, row in enumerate(test.itertuples(index=False)):
            key_gm = (int(row.station_group), int(row.month))
            if n_group_month.get(key_gm, 0) >= min_group_month_samples:
                cond_q[i] = float(q_group_month[key_gm])
                cond_src.append("group_month")
            elif n_group.get(int(row.station_group), 0) >= max(min_group_month_samples, 20):
                cond_q[i] = float(q_group.get(int(row.station_group), q_global))
                cond_src.append("group")
            elif int(row.month) in q_month:
                cond_q[i] = float(q_month[int(row.month)])
                cond_src.append("month")
            else:
                cond_q[i] = q_global
                cond_src.append("global")

        lower_c = test[pred_col].to_numpy() - cond_q
        upper_c = test[pred_col].to_numpy() + cond_q
        coverage_c = float(((test["T_hydromet"].to_numpy() >= lower_c) & (test["T_hydromet"].to_numpy() <= upper_c)).mean())
        width_c = float((upper_c - lower_c).mean())
        wis_cond, crps_cond = interval_quality_stats(
            y_true=test["T_hydromet"].to_numpy(),
            lower=lower_c,
            upper=upper_c,
            target_coverage=float(cov),
        )
        global_rows.append(
            {
                "method": "conditional_station_group_month",
                "target_coverage": cov,
                "achieved_coverage": coverage_c,
                "coverage_gap": coverage_c - cov,
                "mean_width": width_c,
                "wis_mean": wis_cond,
                "crps_like": crps_cond,
            }
        )

        test_cond = test[["station_group", "month", "T_hydromet", pred_col]].copy()
        test_cond["q"] = cond_q
        test_cond["source"] = cond_src
        for (grp, month), g in test_cond.groupby(["station_group", "month"]):
            lo = g[pred_col] - g["q"]
            hi = g[pred_col] + g["q"]
            cov_g = float(((g["T_hydromet"] >= lo) & (g["T_hydromet"] <= hi)).mean())
            wis_g, crps_g = interval_quality_stats(
                y_true=g["T_hydromet"].to_numpy(),
                lower=lo.to_numpy(),
                upper=hi.to_numpy(),
                target_coverage=float(cov),
            )
            src_counts = g["source"].value_counts(dropna=False).to_dict()
            src_main = max(src_counts.items(), key=lambda kv: kv[1])[0]
            conditional_rows.append(
                {
                    "target_coverage": cov,
                    "station_group": int(grp),
                    "month": int(month),
                    "n": int(len(g)),
                    "source_main": str(src_main),
                    "coverage": cov_g,
                    "mean_width": float((hi - lo).mean()),
                    "wis_mean": wis_g,
                    "crps_like": crps_g,
                }
            )

    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows), pd.DataFrame(conditional_rows)


def build_yearly_metrics_table(
    test_df: pd.DataFrame,
    pred_test: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variant, pred in pred_test.items():
        for year, g in test_df.groupby("year"):
            idx = g.index
            pred_y = pd.Series(pred, index=test_df.index).loc[idx].to_numpy()
            rows.append(
                {
                    "variant": variant,
                    "year": int(year),
                    **compute_metrics(g["T_hydromet"], pred_y),
                }
            )
    return pd.DataFrame(rows).sort_values(["variant", "year"]).reset_index(drop=True)


def build_station_risk_summary(
    test_df: pd.DataFrame,
    pred_test: dict[str, np.ndarray],
    baseline_variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_variant not in pred_test:
        return pd.DataFrame(), pd.DataFrame()

    baseline_series = pd.Series(pred_test[baseline_variant], index=test_df.index)
    summary_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[dict[str, float | int | str]] = []

    for variant, pred in pred_test.items():
        if variant == baseline_variant:
            continue
        pred_series = pd.Series(pred, index=test_df.index)
        gains: list[float] = []
        worsened = 0
        improved = 0
        for station, g in test_df.groupby("station"):
            idx = g.index
            mae_b = float(mean_absolute_error(g["T_hydromet"], baseline_series.loc[idx]))
            mae_m = float(mean_absolute_error(g["T_hydromet"], pred_series.loc[idx]))
            gain = mae_b - mae_m
            gains.append(gain)
            worsened += int(gain < 0)
            improved += int(gain > 0)
            detail_rows.append(
                {
                    "variant": variant,
                    "station": str(station),
                    "n_test": int(len(g)),
                    "mae_baseline": mae_b,
                    "mae_model": mae_m,
                    "mae_gain": gain,
                    "worsened": int(gain < 0),
                }
            )
        summary_rows.append(
            {
                "variant": variant,
                "stations_total": int(len(gains)),
                "improved_station_count": int(improved),
                "worsened_station_count": int(worsened),
                "mean_mae_gain": float(np.mean(gains)),
                "median_mae_gain": float(np.median(gains)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_mae_gain", ascending=False).reset_index(drop=True)
    details_df = pd.DataFrame(detail_rows).sort_values(["variant", "mae_gain"]).reset_index(drop=True)
    return summary_df, details_df


def run_delta_gated_for_split(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    n_estimators_override: int | None = None,
) -> dict[str, object]:
    fold_df = pd.concat([train_df, calib_df, test_df], axis=0).copy()
    design, feat_cols = make_design(fold_df)
    X_train = design.loc[train_df.index, feat_cols]
    X_calib = design.loc[calib_df.index, feat_cols]
    X_test = design.loc[test_df.index, feat_cols]
    y_train = train_df["T_hydromet"]

    model = make_xgb_model(args=args, n_estimators_override=n_estimators_override)
    delta_train = y_train.to_numpy() - train_df["T_rp5"].to_numpy()
    model.fit(X_train, pd.Series(delta_train, index=train_df.index))

    pred_calib_global = calib_df["T_rp5"].to_numpy() + model.predict(X_calib)
    pred_test_global = test_df["T_rp5"].to_numpy() + model.predict(X_test)

    calib_eval = calib_df[["station", "T_hydromet", "T_rp5"]].copy()
    calib_eval["pred"] = pred_calib_global
    st_gain = build_station_gain_table(calib_eval, pred_col="pred")
    gate_open = set(st_gain.loc[st_gain["gain"] > args.gate_eps, "station"].astype(str))
    pred_test_gated = np.where(
        test_df["station"].astype(str).isin(gate_open),
        pred_test_global,
        test_df["T_rp5"].to_numpy(),
    )
    baseline = test_df["T_rp5"].to_numpy()
    y = test_df["T_hydromet"]

    baseline_metrics = compute_metrics(y, baseline)
    global_metrics = compute_metrics(y, pred_test_global)
    gated_metrics = compute_metrics(y, pred_test_gated)

    worsened = 0
    improved = 0
    for station, g in test_df.groupby("station"):
        idx = g.index
        pred_st = pd.Series(pred_test_gated, index=test_df.index).loc[idx].to_numpy()
        mae_b = float(mean_absolute_error(g["T_hydromet"], g["T_rp5"]))
        mae_m = float(mean_absolute_error(g["T_hydromet"], pred_st))
        gain = mae_b - mae_m
        worsened += int(gain < 0)
        improved += int(gain > 0)

    return {
        "baseline_metrics": baseline_metrics,
        "xgb_delta_global_metrics": global_metrics,
        "xgb_delta_gated_metrics": gated_metrics,
        "gate_open_station_count": int(len(gate_open)),
        "improved_station_count": int(improved),
        "worsened_station_count": int(worsened),
    }


def run_rolling_origin_diagnostics(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    calib_years = range(args.rolling_calib_start_year, args.rolling_calib_end_year + 1)
    for calib_year in calib_years:
        test_year = calib_year + 1
        train = df[df["year"] < calib_year].copy()
        calib = df[df["year"] == calib_year].copy()
        test = df[df["year"] == test_year].copy()
        if train.empty or calib.empty or test.empty:
            rows.append(
                {
                    "calib_year": int(calib_year),
                    "test_year": int(test_year),
                    "status": "skipped_empty_split",
                }
            )
            continue
        res = run_delta_gated_for_split(
            train_df=train,
            calib_df=calib,
            test_df=test,
            args=args,
            n_estimators_override=args.diag_xgb_n_estimators,
        )
        rows.append(
            {
                "calib_year": int(calib_year),
                "test_year": int(test_year),
                "status": "ok",
                "baseline_RMSE": float(res["baseline_metrics"]["RMSE"]),
                "baseline_MAE": float(res["baseline_metrics"]["MAE"]),
                "xgb_delta_global_RMSE": float(res["xgb_delta_global_metrics"]["RMSE"]),
                "xgb_delta_global_MAE": float(res["xgb_delta_global_metrics"]["MAE"]),
                "xgb_delta_gated_RMSE": float(res["xgb_delta_gated_metrics"]["RMSE"]),
                "xgb_delta_gated_MAE": float(res["xgb_delta_gated_metrics"]["MAE"]),
                "xgb_delta_gated_RMSE_gain_vs_baseline": float(res["baseline_metrics"]["RMSE"] - res["xgb_delta_gated_metrics"]["RMSE"]),
                "xgb_delta_gated_MAE_gain_vs_baseline": float(res["baseline_metrics"]["MAE"] - res["xgb_delta_gated_metrics"]["MAE"]),
                "gate_open_station_count": int(res["gate_open_station_count"]),
                "improved_station_count": int(res["improved_station_count"]),
                "worsened_station_count": int(res["worsened_station_count"]),
                "train_rows": int(len(train)),
                "calib_rows": int(len(calib)),
                "test_rows": int(len(test)),
            }
        )
    return pd.DataFrame(rows)


def run_loso_diagnostics(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base_train = df[df["year"] <= args.train_end_year].copy()
    base_calib = df[df["year"] == args.calib_year].copy()
    base_test = df[(df["year"] >= args.test_start_year) & (df["year"] <= args.test_end_year)].copy()
    stations = base_test.groupby("station").size().sort_values(ascending=False).index.astype(str).tolist()
    if args.loso_max_stations is not None:
        stations = stations[: int(args.loso_max_stations)]

    rows: list[dict[str, float | int | str]] = []
    for st in stations:
        train = base_train[base_train["station"].astype(str) != st].copy()
        calib = base_calib[base_calib["station"].astype(str) != st].copy()
        test = base_test[base_test["station"].astype(str) == st].copy()
        if train.empty or calib.empty or test.empty:
            rows.append({"station": str(st), "status": "skipped_empty_split"})
            continue

        fold_df = pd.concat([train, test], axis=0).copy()
        design, feat_cols = make_design(fold_df)
        X_train = design.loc[train.index, feat_cols]
        X_test = design.loc[test.index, feat_cols]
        y_train = train["T_hydromet"]
        delta_train = y_train.to_numpy() - train["T_rp5"].to_numpy()
        model = make_xgb_model(args=args, n_estimators_override=args.diag_xgb_n_estimators)
        model.fit(X_train, pd.Series(delta_train, index=train.index))
        pred_test = test["T_rp5"].to_numpy() + model.predict(X_test)

        baseline_metrics = compute_metrics(test["T_hydromet"], test["T_rp5"].to_numpy())
        loso_metrics = compute_metrics(test["T_hydromet"], pred_test)
        rows.append(
            {
                "station": str(st),
                "status": "ok",
                "n_test": int(len(test)),
                "baseline_RMSE": float(baseline_metrics["RMSE"]),
                "baseline_MAE": float(baseline_metrics["MAE"]),
                "loso_RMSE": float(loso_metrics["RMSE"]),
                "loso_MAE": float(loso_metrics["MAE"]),
                "loso_RMSE_gain_vs_baseline": float(baseline_metrics["RMSE"] - loso_metrics["RMSE"]),
                "loso_MAE_gain_vs_baseline": float(baseline_metrics["MAE"] - loso_metrics["MAE"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "loso_MAE_gain_vs_baseline"], ascending=[True, False], na_position="last")


def plot_variant_compare(df_metrics: pd.DataFrame, outdir: Path) -> None:
    test = df_metrics[df_metrics["split"] == "test"].copy()
    order = test.sort_values("RMSE")["variant"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = test.set_index("variant").loc[order, "RMSE"]
    ax.bar(np.arange(len(order)), vals.values)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([variant_label_ru(v) for v in order], rotation=30, ha="right")
    ax.set_xlabel("Вариант модели")
    ax.set_ylabel("RMSE, °C")
    ax.set_title("Сравнение вариантов на test (2022-2023): RMSE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "variant_rmse_test.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = test.set_index("variant").loc[order, "MAE"]
    ax.bar(np.arange(len(order)), vals.values)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([variant_label_ru(v) for v in order], rotation=30, ha="right")
    ax.set_xlabel("Вариант модели")
    ax.set_ylabel("MAE, °C")
    ax.set_title("Сравнение вариантов на test (2022-2023): MAE")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "variant_mae_test.png", dpi=140)
    plt.close(fig)


def plot_interval_diagnostics(interval_global: pd.DataFrame, interval_monthly: pd.DataFrame, outdir: Path) -> None:
    method_label = {
        "global_quantile": "Глобальный квантиль",
        "monthly_conformal": "Помесячный conformal",
        "conditional_station_group_month": "Условный conformal (группа станции + месяц)",
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, g in interval_global.groupby("method"):
        gg = g.sort_values("target_coverage")
        ax.plot(
            gg["target_coverage"],
            gg["achieved_coverage"],
            marker="o",
            label=method_label.get(str(method), str(method)),
        )
    ax.plot([0.75, 0.95], [0.75, 0.95], linestyle="--", linewidth=1.1, label="Идеальное совпадение")
    ax.set_xlabel("Целевое покрытие")
    ax.set_ylabel("Фактическое покрытие")
    ax.set_title("Калибровка интервалов неопределённости")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "intervals_target_vs_achieved.png", dpi=140)
    plt.close(fig)

    focus_cov = 0.85
    g = interval_monthly[interval_monthly["target_coverage"] == focus_cov].sort_values("month")
    if len(g):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(g["month"], g["coverage"], marker="o", label="Покрытие по месяцам")
        ax.axhline(focus_cov, linestyle="--", linewidth=1.1, label="Целевое покрытие 0.85")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Покрытие")
        ax.set_title("Покрытие monthly conformal по месяцам")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "intervals_monthly_coverage_085.png", dpi=140)
        plt.close(fig)

    # quality tradeoff for target 0.85: width vs CRPS-like proxy
    q = interval_global[interval_global["target_coverage"] == focus_cov].copy()
    if len(q) and {"mean_width", "crps_like", "method"}.issubset(q.columns):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        qq = q.sort_values("method")
        labels = [method_label.get(str(m), str(m)) for m in qq["method"]]
        x = np.arange(len(qq))

        axes[0].bar(x, qq["mean_width"].to_numpy())
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].set_ylabel("Средняя ширина, °C")
        axes[0].set_title("Ширина интервала (target=0.85)")
        axes[0].grid(axis="y", alpha=0.2)

        axes[1].bar(x, qq["crps_like"].to_numpy())
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].set_ylabel("CRPS-like (меньше лучше)")
        axes[1].set_title("Качество интервала (target=0.85)")
        axes[1].grid(axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(outdir / "intervals_quality_tradeoff_085.png", dpi=140)
        plt.close(fig)


def plot_adaptive_gate_and_safeguard(
    adaptive_gate_df: pd.DataFrame,
    safeguard_policy_df: pd.DataFrame,
    outdir: Path,
) -> None:
    if not adaptive_gate_df.empty and {"gain", "adaptive_eps", "base_eps"}.issubset(adaptive_gate_df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        a = adaptive_gate_df.sort_values("gain", ascending=False).reset_index(drop=True)
        x = np.arange(len(a))
        axes[0].plot(x, a["gain"], linewidth=1.1, label="Gain на calib")
        axes[0].plot(x, a["adaptive_eps"], linewidth=1.1, label="Порог adaptive eps")
        axes[0].axhline(float(a["base_eps"].iloc[0]), linestyle="--", linewidth=1.0, label="Базовый eps")
        axes[0].set_xlabel("Станции (по убыванию gain)")
        axes[0].set_ylabel("MAE gain / eps, °C")
        axes[0].set_title("Adaptive gate по станциям")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        axes[1].hist(a["adaptive_eps"], bins=20, alpha=0.9, label="adaptive eps")
        axes[1].axvline(float(a["base_eps"].iloc[0]), linestyle="--", linewidth=1.0, label="базовый eps")
        axes[1].set_xlabel("eps")
        axes[1].set_ylabel("Частота")
        axes[1].set_title("Распределение station-wise eps")
        axes[1].grid(alpha=0.2)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(outdir / "adaptive_gate_thresholds.png", dpi=140)
        plt.close(fig)

    if not safeguard_policy_df.empty and {"use_fallback", "best_fallback"}.issubset(safeguard_policy_df.columns):
        s = safeguard_policy_df.copy()
        s["use_fallback"] = s["use_fallback"].astype(int)
        fallback_used = s[s["use_fallback"] == 1].copy()
        count_keep = int((s["use_fallback"] == 0).sum())
        count_fb_base = int((fallback_used["best_fallback"] == "baseline").sum())
        count_fb_season = int((fallback_used["best_fallback"] == "seasonal").sum())

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].bar(
            ["Оставлен модельный", "Fallback baseline", "Fallback seasonal"],
            [count_keep, count_fb_base, count_fb_season],
        )
        axes[0].set_ylabel("Число станций")
        axes[0].set_title("Safeguard решения по станциям")
        axes[0].grid(axis="y", alpha=0.2)

        axes[1].hist(s["calib_margin_to_fallback"], bins=30, alpha=0.9, label="mae_model - mae_fallback")
        axes[1].axvline(0.0, linestyle="--", linewidth=1.0, label="граница fallback")
        axes[1].set_xlabel("Разница MAE на calib, °C")
        axes[1].set_ylabel("Частота")
        axes[1].set_title("Запас модели к fallback")
        axes[1].grid(alpha=0.2)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(outdir / "safeguard_policy_summary.png", dpi=140)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    soft_scale_grid = parse_soft_scale_grid(args.soft_scale_grid)
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
    st_df = build_station_gain_table(calib_eval, pred_col="ridge_pred")
    st_df = st_df.rename(columns={"model_mae": "ridge_mae"})
    gate_open = set(st_df.loc[st_df["gain"] > args.gate_eps, "station"].astype(str))
    ridge_station_alpha = {str(r["station"]): float(r["alpha"]) for _, r in st_df.iterrows()}

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

    # 2b) ridge gated by station-month
    calib_sm = calib[["station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_sm["ridge_pred"] = pred_calib["ridge_global"]
    st_month_df = build_station_month_gain_table(
        calib_df=calib_sm,
        pred_col="ridge_pred",
        min_samples=args.min_station_month_samples,
    )
    st_month_df = st_month_df.rename(columns={"model_mae": "ridge_mae"})
    gate_open_sm = set(
        st_month_df.loc[
            (st_month_df["is_eligible"] == 1) & (st_month_df["gain"] > args.gate_eps),
            ["station", "month"],
        ].itertuples(index=False, name=None)
    )
    ridge_station_month_alpha = {
        (str(r["station"]), int(r["month"])): float(r["alpha"])
        for _, r in st_month_df.iterrows()
        if int(r["is_eligible"]) == 1
    }

    pred_calib["ridge_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["ridge_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        open_pairs=gate_open_sm,
    )
    pred_test["ridge_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["ridge_global"],
        station_series=test["station"],
        month_series=test["month"],
        open_pairs=gate_open_sm,
    )

    # 2c) ridge soft blend
    ridge_soft_station_scale, ridge_soft_station_calib = tune_soft_scale_on_calib(
        y_true=y_calib,
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["ridge_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        station_alpha=ridge_station_alpha,
        station_month_alpha=None,
        scale_grid=soft_scale_grid,
    )
    pred_calib["ridge_soft_station"] = ridge_soft_station_calib
    pred_test["ridge_soft_station"] = apply_soft_blend(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["ridge_global"],
        station_series=test["station"],
        month_series=test["month"],
        station_alpha=scale_alpha_map(ridge_station_alpha, ridge_soft_station_scale),
        station_month_alpha=None,
        default_alpha=0.0,
    )

    ridge_soft_station_month_scale, ridge_soft_station_month_calib = tune_soft_scale_on_calib(
        y_true=y_calib,
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["ridge_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        station_alpha=ridge_station_alpha,
        station_month_alpha=ridge_station_month_alpha,
        scale_grid=soft_scale_grid,
    )
    pred_calib["ridge_soft_station_month"] = ridge_soft_station_month_calib
    pred_test["ridge_soft_station_month"] = apply_soft_blend(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["ridge_global"],
        station_series=test["station"],
        month_series=test["month"],
        station_alpha=scale_alpha_map(ridge_station_alpha, ridge_soft_station_month_scale),
        station_month_alpha=scale_alpha_map(ridge_station_month_alpha, ridge_soft_station_month_scale),
        default_alpha=0.0,
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

    # 4b) delta-model: predict delta=(T_hydromet - T_rp5), then restore absolute temperature
    delta_train = y_train.to_numpy() - train["T_rp5"].to_numpy()
    delta_pred_calib = fit_xgb(X_train=X_train, y_train=pd.Series(delta_train, index=train.index), X_pred=X_calib, args=args)
    delta_pred_test = fit_xgb(X_train=X_train, y_train=pd.Series(delta_train, index=train.index), X_pred=X_test, args=args)
    pred_calib["xgb_delta_global"] = calib["T_rp5"].to_numpy() + delta_pred_calib
    pred_test["xgb_delta_global"] = test["T_rp5"].to_numpy() + delta_pred_test

    # 4c) cluster bridge v2: seasonal+bias station profiling (fallback to global if cluster too small)
    station_cluster_map, station_cluster_stats = build_station_cluster_map(
        train_df=train,
        calib_df=calib,
        n_groups=args.cluster_bridge_groups,
        random_state=args.xgb_random_state,
    )
    train_cluster = train["station"].astype(str).map(station_cluster_map).fillna(-1).astype(int)
    calib_cluster = calib["station"].astype(str).map(station_cluster_map).fillna(-1).astype(int)
    test_cluster = test["station"].astype(str).map(station_cluster_map).fillna(-1).astype(int)
    cluster_rows: list[dict[str, float | int | str]] = []

    pred_calib_cluster = pred_calib["xgb_delta_global"].copy()
    pred_test_cluster = pred_test["xgb_delta_global"].copy()
    for cluster_id in sorted(set(train_cluster.tolist())):
        if int(cluster_id) < 0:
            continue
        tr_mask = train_cluster == int(cluster_id)
        tr_n = int(tr_mask.sum())
        calib_mask_c = (calib_cluster == int(cluster_id)).to_numpy()
        test_mask_c = (test_cluster == int(cluster_id)).to_numpy()
        calib_n = int(calib_mask_c.sum())
        test_n = int(test_mask_c.sum())
        if tr_n < args.cluster_bridge_min_train_rows:
            cluster_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "train_rows": tr_n,
                    "calib_rows": calib_n,
                    "test_rows": test_n,
                    "status": "fallback_global",
                }
            )
            continue
        model_cluster = xgb.XGBRegressor(
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
        y_delta_cluster = pd.Series(delta_train, index=train.index).loc[tr_mask]
        model_cluster.fit(X_train.loc[tr_mask], y_delta_cluster)
        if calib_n > 0:
            pred_calib_cluster[calib_mask_c] = (
                calib.loc[calib_mask_c, "T_rp5"].to_numpy() + model_cluster.predict(X_calib.loc[calib_mask_c])
            )
        if test_n > 0:
            pred_test_cluster[test_mask_c] = (
                test.loc[test_mask_c, "T_rp5"].to_numpy() + model_cluster.predict(X_test.loc[test_mask_c])
            )
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "train_rows": tr_n,
                "calib_rows": calib_n,
                "test_rows": test_n,
                "status": "trained",
            }
        )
    pred_calib["xgb_delta_clustered_v2"] = pred_calib_cluster
    pred_test["xgb_delta_clustered_v2"] = pred_test_cluster
    cluster_fit_df = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)

    # 4d) delta-model gated (constant + adaptive risk-aware gate)
    calib_eval_delta = calib[["station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_eval_delta["xgb_delta_pred"] = pred_calib["xgb_delta_global"]
    st_df_delta = build_station_gain_table(calib_eval_delta, pred_col="xgb_delta_pred")
    st_df_delta = st_df_delta.rename(columns={"model_mae": "xgb_delta_mae"})

    # constant gate (historical baseline)
    st_df_delta["base_eps"] = float(args.gate_eps)
    st_df_delta["adaptive_eps"] = float(args.gate_eps)
    pred_calib["xgb_delta_gated"], gate_open_delta = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=st_df_delta,
        model_pred=pred_calib["xgb_delta_global"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )
    pred_test["xgb_delta_gated"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=st_df_delta,
        model_pred=pred_test["xgb_delta_global"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )

    # adaptive gate (station-risk from LOSO + rolling volatility)
    adaptive_gate_df, adaptive_eps_map, adaptive_gate_summary = build_adaptive_gate_table(st_df_delta, args=args)
    pred_calib["xgb_delta_gated_adaptive"], gate_open_delta_adaptive = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=adaptive_gate_df,
        model_pred=pred_calib["xgb_delta_global"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )
    pred_test["xgb_delta_gated_adaptive"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=adaptive_gate_df,
        model_pred=pred_test["xgb_delta_global"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )

    st_month_df_delta = build_station_month_gain_table(
        calib_df=calib_eval_delta,
        pred_col="xgb_delta_pred",
        min_samples=args.min_station_month_samples,
    )
    st_month_df_delta = st_month_df_delta.rename(columns={"model_mae": "xgb_delta_mae"})
    gate_open_sm_delta = set(
        st_month_df_delta.loc[
            (st_month_df_delta["is_eligible"] == 1) & (st_month_df_delta["gain"] > args.gate_eps),
            ["station", "month"],
        ].itertuples(index=False, name=None)
    )
    pred_calib["xgb_delta_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["xgb_delta_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        open_pairs=gate_open_sm_delta,
    )
    pred_test["xgb_delta_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["xgb_delta_global"],
        station_series=test["station"],
        month_series=test["month"],
        open_pairs=gate_open_sm_delta,
    )

    # 4e) safeguard contour for heavy stations (fallback baseline/seasonal if calib worse)
    safeguard_calib_df = calib[["station", "T_hydromet", "T_rp5"]].copy()
    safeguard_calib_df["pred_delta_adaptive"] = pred_calib["xgb_delta_gated_adaptive"]
    safeguard_calib_df["pred_seasonal"] = pred_calib["ridge_seasonal"]
    safeguard_policy_df = build_safeguard_station_policy(
        calib_df=safeguard_calib_df,
        pred_col_model="pred_delta_adaptive",
        pred_col_seasonal="pred_seasonal",
        margin=args.safeguard_margin,
    )
    pred_calib["xgb_delta_gated_adaptive_safeguard"] = apply_safeguard_policy(
        station_series=calib["station"],
        model_pred=pred_calib["xgb_delta_gated_adaptive"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        seasonal_pred=pred_calib["ridge_seasonal"],
        policy_df=safeguard_policy_df,
    )
    pred_test["xgb_delta_gated_adaptive_safeguard"] = apply_safeguard_policy(
        station_series=test["station"],
        model_pred=pred_test["xgb_delta_gated_adaptive"],
        baseline_pred=test["T_rp5"].to_numpy(),
        seasonal_pred=pred_test["ridge_seasonal"],
        policy_df=safeguard_policy_df,
    )

    # 4f) delta-clustered v2 gated (constant and adaptive)
    calib_eval_delta_cluster = calib[["station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_eval_delta_cluster["xgb_delta_cluster_pred"] = pred_calib["xgb_delta_clustered_v2"]
    st_df_delta_cluster = build_station_gain_table(calib_eval_delta_cluster, pred_col="xgb_delta_cluster_pred")
    st_df_delta_cluster = st_df_delta_cluster.rename(columns={"model_mae": "xgb_delta_cluster_mae"})
    st_df_delta_cluster["base_eps"] = float(args.gate_eps)
    st_df_delta_cluster["adaptive_eps"] = st_df_delta_cluster["station"].astype(str).map(adaptive_eps_map).fillna(float(args.gate_eps))

    pred_calib["xgb_delta_clustered_v2_gated"], gate_open_delta_cluster = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=st_df_delta_cluster,
        model_pred=pred_calib["xgb_delta_clustered_v2"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )
    pred_test["xgb_delta_clustered_v2_gated"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=st_df_delta_cluster,
        model_pred=pred_test["xgb_delta_clustered_v2"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )
    pred_calib["xgb_delta_clustered_v2_gated_adaptive"], gate_open_delta_cluster_adaptive = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=st_df_delta_cluster,
        model_pred=pred_calib["xgb_delta_clustered_v2"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )
    pred_test["xgb_delta_clustered_v2_gated_adaptive"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=st_df_delta_cluster,
        model_pred=pred_test["xgb_delta_clustered_v2"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )

    # 4g) cluster bridge v3: seasonal+bias + yearly/mode error profile
    station_cluster_map_v3, station_cluster_stats_v3 = build_station_cluster_map_v3(
        train_df=train,
        calib_df=calib,
        pred_calib_delta_global=pred_calib["xgb_delta_global"],
        n_groups=args.cluster_v3_groups,
        random_state=args.xgb_random_state,
    )
    train_cluster_v3 = train["station"].astype(str).map(station_cluster_map_v3).fillna(-1).astype(int)
    calib_cluster_v3 = calib["station"].astype(str).map(station_cluster_map_v3).fillna(-1).astype(int)
    test_cluster_v3 = test["station"].astype(str).map(station_cluster_map_v3).fillna(-1).astype(int)
    cluster_rows_v3: list[dict[str, float | int | str]] = []

    pred_calib_cluster_v3 = pred_calib["xgb_delta_global"].copy()
    pred_test_cluster_v3 = pred_test["xgb_delta_global"].copy()
    for cluster_id in sorted(set(train_cluster_v3.tolist())):
        if int(cluster_id) < 0:
            continue
        tr_mask = train_cluster_v3 == int(cluster_id)
        tr_n = int(tr_mask.sum())
        calib_mask_c = (calib_cluster_v3 == int(cluster_id)).to_numpy()
        test_mask_c = (test_cluster_v3 == int(cluster_id)).to_numpy()
        calib_n = int(calib_mask_c.sum())
        test_n = int(test_mask_c.sum())
        if tr_n < args.cluster_v3_min_train_rows:
            cluster_rows_v3.append(
                {
                    "cluster_id_v3": int(cluster_id),
                    "train_rows": tr_n,
                    "calib_rows": calib_n,
                    "test_rows": test_n,
                    "status": "fallback_global",
                }
            )
            continue
        model_cluster = make_xgb_model(args=args, n_estimators_override=None)
        y_delta_cluster = pd.Series(delta_train, index=train.index).loc[tr_mask]
        model_cluster.fit(X_train.loc[tr_mask], y_delta_cluster)
        if calib_n > 0:
            pred_calib_cluster_v3[calib_mask_c] = (
                calib.loc[calib_mask_c, "T_rp5"].to_numpy() + model_cluster.predict(X_calib.loc[calib_mask_c])
            )
        if test_n > 0:
            pred_test_cluster_v3[test_mask_c] = (
                test.loc[test_mask_c, "T_rp5"].to_numpy() + model_cluster.predict(X_test.loc[test_mask_c])
            )
        cluster_rows_v3.append(
            {
                "cluster_id_v3": int(cluster_id),
                "train_rows": tr_n,
                "calib_rows": calib_n,
                "test_rows": test_n,
                "status": "trained",
            }
        )
    pred_calib["xgb_delta_clustered_v3"] = pred_calib_cluster_v3
    pred_test["xgb_delta_clustered_v3"] = pred_test_cluster_v3
    cluster_fit_df_v3 = pd.DataFrame(cluster_rows_v3).sort_values("cluster_id_v3").reset_index(drop=True)

    # 4h) cluster v3 gated (constant and adaptive)
    calib_eval_delta_cluster_v3 = calib[["station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_eval_delta_cluster_v3["xgb_delta_cluster_v3_pred"] = pred_calib["xgb_delta_clustered_v3"]
    st_df_delta_cluster_v3 = build_station_gain_table(calib_eval_delta_cluster_v3, pred_col="xgb_delta_cluster_v3_pred")
    st_df_delta_cluster_v3 = st_df_delta_cluster_v3.rename(columns={"model_mae": "xgb_delta_cluster_v3_mae"})
    st_df_delta_cluster_v3["base_eps"] = float(args.gate_eps)
    st_df_delta_cluster_v3["adaptive_eps"] = st_df_delta_cluster_v3["station"].astype(str).map(adaptive_eps_map).fillna(float(args.gate_eps))

    pred_calib["xgb_delta_clustered_v3_gated"], gate_open_delta_cluster_v3 = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=st_df_delta_cluster_v3,
        model_pred=pred_calib["xgb_delta_clustered_v3"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )
    pred_test["xgb_delta_clustered_v3_gated"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=st_df_delta_cluster_v3,
        model_pred=pred_test["xgb_delta_clustered_v3"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="base_eps",
    )
    pred_calib["xgb_delta_clustered_v3_gated_adaptive"], gate_open_delta_cluster_v3_adaptive = apply_station_gate_with_eps(
        station_series=calib["station"],
        gain_table=st_df_delta_cluster_v3,
        model_pred=pred_calib["xgb_delta_clustered_v3"],
        baseline_pred=calib["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )
    pred_test["xgb_delta_clustered_v3_gated_adaptive"], _ = apply_station_gate_with_eps(
        station_series=test["station"],
        gain_table=st_df_delta_cluster_v3,
        model_pred=pred_test["xgb_delta_clustered_v3"],
        baseline_pred=test["T_rp5"].to_numpy(),
        eps_col="adaptive_eps",
    )

    # 5) xgb gated by station
    calib_eval_x = calib[["station", "T_hydromet", "T_rp5"]].copy()
    calib_eval_x["xgb_pred"] = pred_calib["xgb_global"]
    st_df_x = build_station_gain_table(calib_eval_x, pred_col="xgb_pred")
    st_df_x = st_df_x.rename(columns={"model_mae": "xgb_mae"})
    gate_open_x = set(st_df_x.loc[st_df_x["gain"] > args.gate_eps, "station"].astype(str))
    xgb_station_alpha = {str(r["station"]): float(r["alpha"]) for _, r in st_df_x.iterrows()}

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

    # 5b) xgb gated by station-month
    calib_sm_x = calib[["station", "month", "T_hydromet", "T_rp5"]].copy()
    calib_sm_x["xgb_pred"] = pred_calib["xgb_global"]
    st_month_df_x = build_station_month_gain_table(
        calib_df=calib_sm_x,
        pred_col="xgb_pred",
        min_samples=args.min_station_month_samples,
    )
    st_month_df_x = st_month_df_x.rename(columns={"model_mae": "xgb_mae"})
    gate_open_sm_x = set(
        st_month_df_x.loc[
            (st_month_df_x["is_eligible"] == 1) & (st_month_df_x["gain"] > args.gate_eps),
            ["station", "month"],
        ].itertuples(index=False, name=None)
    )
    xgb_station_month_alpha = {
        (str(r["station"]), int(r["month"])): float(r["alpha"])
        for _, r in st_month_df_x.iterrows()
        if int(r["is_eligible"]) == 1
    }

    pred_calib["xgb_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["xgb_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        open_pairs=gate_open_sm_x,
    )
    pred_test["xgb_gated_station_month"] = apply_hard_gate_station_month(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["xgb_global"],
        station_series=test["station"],
        month_series=test["month"],
        open_pairs=gate_open_sm_x,
    )

    # 5c) xgb soft blend
    xgb_soft_station_scale, xgb_soft_station_calib = tune_soft_scale_on_calib(
        y_true=y_calib,
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["xgb_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        station_alpha=xgb_station_alpha,
        station_month_alpha=None,
        scale_grid=soft_scale_grid,
    )
    pred_calib["xgb_soft_station"] = xgb_soft_station_calib
    pred_test["xgb_soft_station"] = apply_soft_blend(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["xgb_global"],
        station_series=test["station"],
        month_series=test["month"],
        station_alpha=scale_alpha_map(xgb_station_alpha, xgb_soft_station_scale),
        station_month_alpha=None,
        default_alpha=0.0,
    )

    xgb_soft_station_month_scale, xgb_soft_station_month_calib = tune_soft_scale_on_calib(
        y_true=y_calib,
        baseline_pred=calib["T_rp5"].to_numpy(),
        model_pred=pred_calib["xgb_global"],
        station_series=calib["station"],
        month_series=calib["month"],
        station_alpha=xgb_station_alpha,
        station_month_alpha=xgb_station_month_alpha,
        scale_grid=soft_scale_grid,
    )
    pred_calib["xgb_soft_station_month"] = xgb_soft_station_month_calib
    pred_test["xgb_soft_station_month"] = apply_soft_blend(
        baseline_pred=test["T_rp5"].to_numpy(),
        model_pred=pred_test["xgb_global"],
        station_series=test["station"],
        month_series=test["month"],
        station_alpha=scale_alpha_map(xgb_station_alpha, xgb_soft_station_month_scale),
        station_month_alpha=scale_alpha_map(xgb_station_month_alpha, xgb_soft_station_month_scale),
        default_alpha=0.0,
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
    test_rank = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] != "baseline")].copy()
    best_variant_test_rmse = str(test_rank.sort_values("RMSE").iloc[0]["variant"])
    best_variant_test_mae = str(test_rank.sort_values("MAE").iloc[0]["variant"])

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
    interval_global, interval_monthly, interval_conditional = build_conformal_intervals(
        calib_diag,
        test_diag,
        pred_col="pred_best",
        station_groups=args.conformal_station_groups,
        min_group_month_samples=args.conformal_min_group_month_samples,
    )
    interval_global.to_csv(outdir / "intervals_summary.csv", index=False)
    interval_global.to_csv(outdir / "intervals_quality_summary.csv", index=False)
    interval_monthly.to_csv(outdir / "intervals_by_month.csv", index=False)
    interval_conditional.to_csv(outdir / "intervals_by_station_group_month.csv", index=False)

    # robustness diagnostics
    yearly_metrics_df = build_yearly_metrics_table(test_df=test, pred_test=pred_test)
    yearly_metrics_df.to_csv(outdir / "metrics_by_test_year.csv", index=False)
    risk_summary_df, risk_details_df = build_station_risk_summary(test_df=test, pred_test=pred_test, baseline_variant="baseline")
    risk_summary_df.to_csv(outdir / "station_risk_summary_test.csv", index=False)
    risk_details_df.to_csv(outdir / "station_risk_details_test.csv", index=False)
    rolling_df = pd.DataFrame()
    loso_df = pd.DataFrame()
    if args.run_rolling_origin:
        rolling_df = run_rolling_origin_diagnostics(df=df, args=args)
        rolling_df.to_csv(outdir / "rolling_origin_summary.csv", index=False)
    if args.run_loso:
        loso_df = run_loso_diagnostics(df=df, args=args)
        loso_df.to_csv(outdir / "loso_summary.csv", index=False)

    # plots
    plot_variant_compare(metrics_df, outdir)
    plot_interval_diagnostics(interval_global, interval_monthly, outdir)
    plot_adaptive_gate_and_safeguard(
        adaptive_gate_df=adaptive_gate_df,
        safeguard_policy_df=safeguard_policy_df,
        outdir=outdir,
    )

    # save gating / heavy lists
    pd.DataFrame({"station": sorted(gate_open)}).to_csv(outdir / "ridge_gated_open_stations.csv", index=False)
    pd.DataFrame({"station": sorted(gate_open_x)}).to_csv(outdir / "xgb_gated_open_stations.csv", index=False)
    pd.DataFrame(sorted(gate_open_sm), columns=["station", "month"]).to_csv(
        outdir / "ridge_station_month_gated_open.csv", index=False
    )
    pd.DataFrame(sorted(gate_open_sm_x), columns=["station", "month"]).to_csv(
        outdir / "xgb_station_month_gated_open.csv", index=False
    )
    pd.DataFrame({"station": sorted(gate_open_delta)}).to_csv(outdir / "xgb_delta_gated_open_stations.csv", index=False)
    pd.DataFrame({"station": sorted(gate_open_delta_adaptive)}).to_csv(
        outdir / "xgb_delta_gated_adaptive_open_stations.csv", index=False
    )
    pd.DataFrame(sorted(gate_open_sm_delta), columns=["station", "month"]).to_csv(
        outdir / "xgb_delta_station_month_gated_open.csv", index=False
    )
    pd.DataFrame({"station": sorted(gate_open_delta_cluster)}).to_csv(
        outdir / "xgb_delta_clustered_v2_gated_open_stations.csv", index=False
    )
    pd.DataFrame({"station": sorted(gate_open_delta_cluster_adaptive)}).to_csv(
        outdir / "xgb_delta_clustered_v2_gated_adaptive_open_stations.csv", index=False
    )
    pd.DataFrame({"station": sorted(gate_open_delta_cluster_v3)}).to_csv(
        outdir / "xgb_delta_clustered_v3_gated_open_stations.csv", index=False
    )
    pd.DataFrame({"station": sorted(gate_open_delta_cluster_v3_adaptive)}).to_csv(
        outdir / "xgb_delta_clustered_v3_gated_adaptive_open_stations.csv", index=False
    )
    pd.DataFrame({"station": sorted(heavy_stations)}).to_csv(outdir / "ridge_heavy_stations.csv", index=False)
    adaptive_gate_df.to_csv(outdir / "xgb_delta_adaptive_gate_thresholds.csv", index=False)
    safeguard_policy_df.to_csv(outdir / "xgb_delta_adaptive_safeguard_policy.csv", index=False)
    save_json(outdir / "adaptive_gate_summary.json", adaptive_gate_summary)
    save_json(
        outdir / "safeguard_summary.json",
        {
            "margin": float(args.safeguard_margin),
            "stations_total": int(len(safeguard_policy_df)),
            "fallback_station_count": int((safeguard_policy_df["use_fallback"] == 1).sum()),
            "fallback_baseline_count": int(
                ((safeguard_policy_df["use_fallback"] == 1) & (safeguard_policy_df["best_fallback"] == "baseline")).sum()
            ),
            "fallback_seasonal_count": int(
                ((safeguard_policy_df["use_fallback"] == 1) & (safeguard_policy_df["best_fallback"] == "seasonal")).sum()
            ),
        },
    )
    st_df.sort_values("gain", ascending=False).to_csv(outdir / "ridge_station_gain_on_calib.csv", index=False)
    st_df_x.sort_values("gain", ascending=False).to_csv(outdir / "xgb_station_gain_on_calib.csv", index=False)
    st_month_df.sort_values("gain", ascending=False).to_csv(outdir / "ridge_station_month_gain_on_calib.csv", index=False)
    st_month_df_x.sort_values("gain", ascending=False).to_csv(outdir / "xgb_station_month_gain_on_calib.csv", index=False)
    st_df_delta.sort_values("gain", ascending=False).to_csv(outdir / "xgb_delta_station_gain_on_calib.csv", index=False)
    st_month_df_delta.sort_values("gain", ascending=False).to_csv(
        outdir / "xgb_delta_station_month_gain_on_calib.csv", index=False
    )
    st_df_delta_cluster.sort_values("gain", ascending=False).to_csv(
        outdir / "xgb_delta_clustered_v2_station_gain_on_calib.csv", index=False
    )
    st_df_delta_cluster_v3.sort_values("gain", ascending=False).to_csv(
        outdir / "xgb_delta_clustered_v3_station_gain_on_calib.csv", index=False
    )
    station_cluster_stats.sort_values(["cluster_id", "station"]).to_csv(
        outdir / "station_cluster_stats_train.csv", index=False
    )
    pd.DataFrame(
        {"station": list(station_cluster_map.keys()), "cluster_id": list(station_cluster_map.values())}
    ).sort_values(["cluster_id", "station"]).to_csv(outdir / "station_cluster_map.csv", index=False)
    cluster_fit_df.to_csv(outdir / "cluster_fit_status.csv", index=False)
    station_cluster_stats_v3.sort_values(["cluster_id_v3", "station"]).to_csv(
        outdir / "station_cluster_stats_train_v3.csv", index=False
    )
    pd.DataFrame(
        {"station": list(station_cluster_map_v3.keys()), "cluster_id_v3": list(station_cluster_map_v3.values())}
    ).sort_values(["cluster_id_v3", "station"]).to_csv(outdir / "station_cluster_map_v3.csv", index=False)
    cluster_fit_df_v3.to_csv(outdir / "cluster_fit_status_v3.csv", index=False)

    # summary json
    best_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == best_variant)].iloc[0].to_dict()
    best_test_rmse = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == best_variant_test_rmse)].iloc[0].to_dict()
    best_test_mae = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == best_variant_test_mae)].iloc[0].to_dict()
    base_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["variant"] == "baseline")].iloc[0].to_dict()
    xgb_delta_gated_risk = (
        risk_summary_df[risk_summary_df["variant"] == "xgb_delta_gated"].iloc[0].to_dict()
        if not risk_summary_df.empty and (risk_summary_df["variant"] == "xgb_delta_gated").any()
        else {}
    )
    xgb_delta_gated_adaptive_risk = (
        risk_summary_df[risk_summary_df["variant"] == "xgb_delta_gated_adaptive"].iloc[0].to_dict()
        if not risk_summary_df.empty and (risk_summary_df["variant"] == "xgb_delta_gated_adaptive").any()
        else {}
    )
    xgb_delta_gated_adaptive_safeguard_risk = (
        risk_summary_df[risk_summary_df["variant"] == "xgb_delta_gated_adaptive_safeguard"].iloc[0].to_dict()
        if not risk_summary_df.empty and (risk_summary_df["variant"] == "xgb_delta_gated_adaptive_safeguard").any()
        else {}
    )
    xgb_delta_clustered_v2_gated_risk = (
        risk_summary_df[risk_summary_df["variant"] == "xgb_delta_clustered_v2_gated"].iloc[0].to_dict()
        if not risk_summary_df.empty and (risk_summary_df["variant"] == "xgb_delta_clustered_v2_gated").any()
        else {}
    )
    xgb_delta_clustered_v3_gated_risk = (
        risk_summary_df[risk_summary_df["variant"] == "xgb_delta_clustered_v3_gated"].iloc[0].to_dict()
        if not risk_summary_df.empty and (risk_summary_df["variant"] == "xgb_delta_clustered_v3_gated").any()
        else {}
    )
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
        "best_variant_by_test_rmse": best_variant_test_rmse,
        "best_variant_by_test_mae": best_variant_test_mae,
        "best_test_metrics": best_test,
        "best_test_rmse_metrics": best_test_rmse,
        "best_test_mae_metrics": best_test_mae,
        "baseline_test_metrics": base_test,
        "best_minus_baseline_test": {
            "RMSE_delta": float(best_test["RMSE"] - base_test["RMSE"]),
            "MAE_delta": float(best_test["MAE"] - base_test["MAE"]),
            "R2_delta": float(best_test["R2"] - base_test["R2"]),
        },
        "ridge_gated_open_station_count": int(len(gate_open)),
        "ridge_gated_open_station_month_count": int(len(gate_open_sm)),
        "xgb_gated_open_station_count": int(len(gate_open_x)),
        "xgb_gated_open_station_month_count": int(len(gate_open_sm_x)),
        "xgb_delta_gated_open_station_count": int(len(gate_open_delta)),
        "xgb_delta_gated_adaptive_open_station_count": int(len(gate_open_delta_adaptive)),
        "xgb_delta_gated_open_station_month_count": int(len(gate_open_sm_delta)),
        "xgb_delta_clustered_v2_gated_open_station_count": int(len(gate_open_delta_cluster)),
        "xgb_delta_clustered_v2_gated_adaptive_open_station_count": int(len(gate_open_delta_cluster_adaptive)),
        "xgb_delta_clustered_v3_gated_open_station_count": int(len(gate_open_delta_cluster_v3)),
        "xgb_delta_clustered_v3_gated_adaptive_open_station_count": int(len(gate_open_delta_cluster_v3_adaptive)),
        "cluster_bridge_groups": int(args.cluster_bridge_groups),
        "cluster_bridge_min_train_rows": int(args.cluster_bridge_min_train_rows),
        "cluster_bridge_trained_cluster_count": int((cluster_fit_df["status"] == "trained").sum()),
        "cluster_bridge_fallback_cluster_count": int((cluster_fit_df["status"] != "trained").sum()),
        "cluster_v3_groups": int(args.cluster_v3_groups),
        "cluster_v3_min_train_rows": int(args.cluster_v3_min_train_rows),
        "cluster_v3_trained_cluster_count": int((cluster_fit_df_v3["status"] == "trained").sum()),
        "cluster_v3_fallback_cluster_count": int((cluster_fit_df_v3["status"] != "trained").sum()),
        "ridge_heavy_station_count": int(len(heavy_stations)),
        "min_station_month_samples": int(args.min_station_month_samples),
        "conformal_station_groups": int(args.conformal_station_groups),
        "conformal_min_group_month_samples": int(args.conformal_min_group_month_samples),
        "adaptive_gate_summary": adaptive_gate_summary,
        "safeguard_summary": {
            "margin": float(args.safeguard_margin),
            "stations_total": int(len(safeguard_policy_df)),
            "fallback_station_count": int((safeguard_policy_df["use_fallback"] == 1).sum()),
            "fallback_baseline_count": int(
                ((safeguard_policy_df["use_fallback"] == 1) & (safeguard_policy_df["best_fallback"] == "baseline")).sum()
            ),
            "fallback_seasonal_count": int(
                ((safeguard_policy_df["use_fallback"] == 1) & (safeguard_policy_df["best_fallback"] == "seasonal")).sum()
            ),
        },
        "station_risk_summary": {
            "xgb_delta_gated": xgb_delta_gated_risk,
            "xgb_delta_gated_adaptive": xgb_delta_gated_adaptive_risk,
            "xgb_delta_gated_adaptive_safeguard": xgb_delta_gated_adaptive_safeguard_risk,
            "xgb_delta_clustered_v2_gated": xgb_delta_clustered_v2_gated_risk,
            "xgb_delta_clustered_v3_gated": xgb_delta_clustered_v3_gated_risk,
        },
        "rolling_origin_enabled": bool(args.run_rolling_origin),
        "loso_enabled": bool(args.run_loso),
        "rolling_origin_rows": int(len(rolling_df)),
        "loso_rows": int(len(loso_df)),
        "recommended_runtime_config": {
            "gate_eps": float(args.gate_eps),
            "adaptive_gate_enabled": bool(args.adaptive_gate_enabled),
            "conformal_station_groups": int(args.conformal_station_groups),
            "conformal_min_group_month_samples": int(args.conformal_min_group_month_samples),
        },
        "soft_scale_grid": soft_scale_grid,
        "selected_soft_scales": {
            "ridge_soft_station": float(ridge_soft_station_scale),
            "ridge_soft_station_month": float(ridge_soft_station_month_scale),
            "xgb_soft_station": float(xgb_soft_station_scale),
            "xgb_soft_station_month": float(xgb_soft_station_month_scale),
        },
    }
    save_json(outdir / "summary.json", summary)

    print(f"Saved improvement run: {outdir}")
    print(f"Best variant by calib MAE: {best_variant}")
    print(f"Best variant by test RMSE: {best_variant_test_rmse}")
    print(f"Best variant by test MAE: {best_variant_test_mae}")
    print(
        "Test delta vs baseline:",
        f"RMSE={summary['best_minus_baseline_test']['RMSE_delta']:.6f},",
        f"MAE={summary['best_minus_baseline_test']['MAE_delta']:.6f},",
        f"R2={summary['best_minus_baseline_test']['R2_delta']:.6f}",
    )


if __name__ == "__main__":
    main()
