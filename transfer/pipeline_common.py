from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

DATE_CANDIDATES = ("date", "Date", "datetime", "dt", "timestamp", "time")
STATION_CANDIDATES = ("station_id", "station", "Cod", "code", "station_code", "stationid")
LAT_CANDIDATES = ("lat", "latitude", "Latitude", "LAT", "Y_final", "y", "Y")
LON_CANDIDATES = ("lon", "longitude", "Longitude", "LON", "X_final", "x", "X")
TARGET_COLUMN = "T"
LAG_BASE_COLUMNS = ("Temperature_2m", "Dewpoint_2m", "LST_Day", "LST_Night")
BASE_FEATURE_ORDER = (
    "Temperature_2m",
    "Dewpoint_2m",
    "Surface_pressure",
    "Evaporation",
    "Total_precipitation",
    "LST_Day",
    "LST_Night",
    "dayofyear",
    "sin_doy",
    "cos_doy",
    "dewpoint_dep",
    "diurnal_range",
    "precip_any",
    "precip_log1p",
    "year",
    "month",
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "station_train_mean_T",
)


@dataclass(frozen=True)
class DatasetMeta:
    date_col: str
    station_col: str
    lat_col: str | None
    lon_col: str | None
    target_col: str = TARGET_COLUMN


def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def infer_column(df: pd.DataFrame, candidates: Sequence[str], fallback: str | None = None) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return fallback


def infer_meta(df: pd.DataFrame) -> DatasetMeta:
    date_col = infer_column(df, DATE_CANDIDATES)
    if date_col is None:
        raise RuntimeError("Не удалось определить колонку даты")

    station_col = infer_column(df, STATION_CANDIDATES, "__station__")
    lat_col = infer_column(df, LAT_CANDIDATES)
    lon_col = infer_column(df, LON_CANDIDATES)
    return DatasetMeta(date_col=date_col, station_col=station_col, lat_col=lat_col, lon_col=lon_col)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetMeta]:
    df = pd.read_csv(path)
    meta = infer_meta(df)

    if meta.station_col not in df.columns:
        df[meta.station_col] = 0

    df[meta.date_col] = pd.to_datetime(df[meta.date_col])
    return df, meta


def add_calendar_features(df: pd.DataFrame, meta: DatasetMeta) -> pd.DataFrame:
    out = df.copy()
    if "year" not in out.columns:
        out["year"] = out[meta.date_col].dt.year
    if "month" not in out.columns:
        out["month"] = out[meta.date_col].dt.month
    if "dayofyear" not in out.columns:
        out["dayofyear"] = out[meta.date_col].dt.dayofyear
    if "sin_doy" not in out.columns:
        out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    if "cos_doy" not in out.columns:
        out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def add_derived_features(df: pd.DataFrame, zero_inflated_precip: bool = False) -> pd.DataFrame:
    out = df.copy()
    if "dewpoint_dep" not in out.columns and {"Temperature_2m", "Dewpoint_2m"}.issubset(out.columns):
        out["dewpoint_dep"] = out["Temperature_2m"] - out["Dewpoint_2m"]
    if "diurnal_range" not in out.columns and {"LST_Day", "LST_Night"}.issubset(out.columns):
        out["diurnal_range"] = out["LST_Day"] - out["LST_Night"]
    if zero_inflated_precip and "Total_precipitation" in out.columns:
        p = out["Total_precipitation"].fillna(0.0).clip(lower=0.0)
        out["precip_any"] = (p > 0).astype(np.int8)
        out["precip_log1p"] = np.log1p(p)
    return out


def add_spatial_features(df: pd.DataFrame, meta: DatasetMeta) -> pd.DataFrame:
    out = df.copy()
    if meta.lat_col and meta.lon_col:
        latv = out[meta.lat_col].astype(float)
        lonv = out[meta.lon_col].astype(float)
        plausible_deg = latv.abs().max() <= 90 and lonv.abs().max() <= 180
        if plausible_deg:
            latr = np.deg2rad(latv)
            lonr = np.deg2rad(lonv)
            out["sin_lat"] = np.sin(latr)
            out["cos_lat"] = np.cos(latr)
            out["sin_lon"] = np.sin(lonr)
            out["cos_lon"] = np.cos(lonr)
        else:
            x = (lonv - lonv.min()) / (lonv.max() - lonv.min() + 1e-9)
            y = (latv - latv.min()) / (latv.max() - latv.min() + 1e-9)
            out["sin_lat"] = np.sin(2 * np.pi * y)
            out["cos_lat"] = np.cos(2 * np.pi * y)
            out["sin_lon"] = np.sin(2 * np.pi * x)
            out["cos_lon"] = np.cos(2 * np.pi * x)
    else:
        out["sin_lat"] = 0.0
        out["cos_lat"] = 1.0
        out["sin_lon"] = 0.0
        out["cos_lon"] = 1.0
    return out


def add_lag_features(
    df: pd.DataFrame,
    meta: DatasetMeta,
    lag_steps: Iterable[int] = (1, 2, 3),
    lag_columns: Sequence[str] = LAG_BASE_COLUMNS,
) -> pd.DataFrame:
    out = df.sort_values([meta.station_col, meta.date_col]).copy()
    for col in lag_columns:
        if col not in out.columns:
            continue
        for lag in lag_steps:
            out[f"{col}_lag{lag}"] = out.groupby(meta.station_col)[col].shift(lag)
    return out


def add_station_train_mean_feature(
    df: pd.DataFrame,
    meta: DatasetMeta,
    train_mask: pd.Series,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    out = df.copy()
    train_mean = (
        out.loc[train_mask].dropna(subset=[target_col]).groupby(meta.station_col)[target_col].mean().rename("station_train_mean_T")
    )
    out = out.merge(train_mean, left_on=meta.station_col, right_index=True, how="left")
    out["station_train_mean_T"] = out["station_train_mean_T"].fillna(out["station_train_mean_T"].mean())
    return out


def build_feature_frame(
    df: pd.DataFrame,
    meta: DatasetMeta,
    train_mask: pd.Series | None = None,
    lag_steps: Iterable[int] = (1, 2, 3),
    zero_inflated_precip: bool = False,
    include_station_mean: bool = True,
) -> pd.DataFrame:
    out = add_calendar_features(df, meta)
    out = add_derived_features(out, zero_inflated_precip=zero_inflated_precip)
    out = add_spatial_features(out, meta)
    out = add_lag_features(out, meta, lag_steps=lag_steps)
    if include_station_mean:
        if train_mask is None:
            raise ValueError("train_mask обязателен, если include_station_mean=True")
        out = add_station_train_mean_feature(out, meta, train_mask=train_mask)
    return out


def resolve_feature_list(
    df: pd.DataFrame,
    lag_steps: Iterable[int] = (1, 2, 3),
    include_station_mean: bool = True,
) -> list[str]:
    features = list(BASE_FEATURE_ORDER)
    if not include_station_mean:
        features = [col for col in features if col != "station_train_mean_T"]
    lag_features = [f"{col}_lag{lag}" for col in LAG_BASE_COLUMNS for lag in lag_steps]
    return [col for col in features + lag_features if col in df.columns]


def split_by_year(
    df: pd.DataFrame,
    train_start_year: int = 2013,
    train_end_year: int = 2021,
    test_start_year: int = 2022,
    test_end_year: int = 2023,
) -> tuple[pd.Series, pd.Series]:
    train_mask = (df["year"] >= train_start_year) & (df["year"] <= train_end_year)
    test_mask = (df["year"] >= test_start_year) & (df["year"] <= test_end_year)
    return train_mask, test_mask


def filter_winter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["month"].isin([11, 12, 1, 2, 3])].copy()


def limit_to_station_subset(df: pd.DataFrame, station_col: str, max_stations: int | None, seed: int = 42) -> pd.DataFrame:
    if max_stations is None:
        return df
    unique_stations = pd.Series(sorted(df[station_col].dropna().unique()))
    if len(unique_stations) <= max_stations:
        return df
    keep = unique_stations.sample(max_stations, random_state=seed).tolist()
    return df[df[station_col].isin(keep)].copy()


def make_dmatrix(df: pd.DataFrame, features: Sequence[str], target_col: str = TARGET_COLUMN) -> xgb.DMatrix:
    return xgb.DMatrix(df[features], label=df[target_col])


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def save_json(path: str | Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    output_csv: str | Path,
) -> None:
    rows: list[dict[str, float | int]] = []
    for group_key, group_df in df.assign(__y=y_true, __p=y_pred).groupby(group_col):
        if len(group_df) < 5:
            continue
        rows.append(
            {
                "group": group_key,
                "n": int(len(group_df)),
                "R2": float(r2_score(group_df["__y"], group_df["__p"])) if len(group_df) >= 2 else np.nan,
                "RMSE": float(np.sqrt(mean_squared_error(group_df["__y"], group_df["__p"]))),
                "MAE": float(mean_absolute_error(group_df["__y"], group_df["__p"])),
                "MedAE": float(median_absolute_error(group_df["__y"], group_df["__p"])),
            }
        )
    if not rows:
        pd.DataFrame(columns=["group", "n", "R2", "RMSE", "MAE", "MedAE"]).to_csv(output_csv, index=False)
        return
    pd.DataFrame(rows).sort_values("group").to_csv(output_csv, index=False)


def save_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, output_png: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    plt.scatter(y_true, y_pred, s=4, alpha=0.5)
    plt.plot(lims, lims, lw=2)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_residual_hist(residuals: np.ndarray, output_png: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=60)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_density_plot(y_true: np.ndarray, y_pred: np.ndarray, output_png: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x = y_true[idx]
        y = y_pred[idx]
        plt.scatter(x, y, c=z[idx], s=3)
    except Exception:
        plt.scatter(y_true, y_pred, s=3, alpha=0.3)
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    plt.plot(lims, lims, lw=2)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def tune_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Sequence[str],
    *,
    target_col: str = TARGET_COLUMN,
    device: str = "cuda",
    n_trials: int = 40,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 150,
    seed: int = 42,
    learning_rate_low: float = 0.005,
    learning_rate_high: float = 0.10,
) -> dict[str, float | int]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    dtrain = make_dmatrix(train_df, features, target_col=target_col)
    dval = make_dmatrix(val_df, features, target_col=target_col)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": device,
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", learning_rate_low, learning_rate_high, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "seed": seed,
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        pred = model.predict(dval)
        return r2_score(val_df[target_col], pred)

    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    best_params.update({"objective": "reg:squarederror", "tree_method": "hist", "device": device, "seed": seed})
    return best_params


def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Sequence[str],
    params: dict[str, float | int],
    *,
    target_col: str = TARGET_COLUMN,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 150,
    base_model: xgb.Booster | str | None = None,
    verbose_eval: bool | int = False,
) -> xgb.Booster:
    dtrain = make_dmatrix(train_df, features, target_col=target_col)
    dval = make_dmatrix(val_df, features, target_col=target_col)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        xgb_model=base_model,
        verbose_eval=verbose_eval,
    )


def evaluate_model(
    model: xgb.Booster,
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    target_col: str = TARGET_COLUMN,
) -> tuple[np.ndarray, dict[str, float | int]]:
    preds = model.predict(make_dmatrix(df, features, target_col=target_col))
    return preds, compute_metrics(df[target_col], preds)


def choose_validation_year(train_df: pd.DataFrame) -> int:
    return int(train_df["year"].max())


def save_run_bundle(
    outdir: str | Path,
    *,
    metrics: dict[str, dict[str, float | int]],
    features: Sequence[str],
    params: dict[str, float | int] | None = None,
    predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] | None = None,
    model: xgb.Booster | None = None,
    station_col: str | None = None,
) -> None:
    outdir = ensure_dir(outdir)
    save_json(Path(outdir) / "features_used.json", list(features))
    for split_name, split_metrics in metrics.items():
        save_json(Path(outdir) / f"metrics_{split_name}.json", split_metrics)
    if params is not None:
        save_json(Path(outdir) / "params.json", params)
    if model is not None:
        model.save_model(str(Path(outdir) / "model.json"))
    if predictions:
        for split_name, (split_df, preds) in predictions.items():
            split_df = split_df.copy()
            split_df["y_pred"] = preds
            split_df.to_csv(Path(outdir) / f"predictions_{split_name}.csv", index=False)
            save_scatter_plot(split_df[TARGET_COLUMN].to_numpy(), preds, Path(outdir) / f"scatter_{split_name}.png", f"Pred vs True ({split_name})")
            save_residual_hist(preds - split_df[TARGET_COLUMN].to_numpy(), Path(outdir) / f"residuals_{split_name}.png", f"Residuals ({split_name})")
            if split_name in {"test", "full"}:
                save_density_plot(split_df[TARGET_COLUMN].to_numpy(), preds, Path(outdir) / f"density_{split_name}.png", f"Density True vs Pred ({split_name})")
            if "month" in split_df.columns:
                save_group_metrics(split_df, "month", split_df[TARGET_COLUMN], preds, Path(outdir) / f"metrics_by_month_{split_name}.csv")
            if station_col and station_col in split_df.columns:
                save_group_metrics(split_df, station_col, split_df[TARGET_COLUMN], preds, Path(outdir) / f"metrics_by_station_{split_name}.csv")
