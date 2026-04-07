from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from pipeline_common import ensure_dir, save_json

DATE_CANDIDATES = ("Date", "date", "datetime", "timestamp")
STATION_CANDIDATES = ("station", "station_id", "Cod", "code", "station_code")
RP5_TEMP_CANDIDATES = ("T_rp5", "rp5_T", "rp5_temp", "Temperature_rp5", "T_rp5_raw")
HYDROMET_TEMP_CANDIDATES = ("T_hydromet", "hydromet_T", "official_T", "T_official", "T_rosgidromet")


def infer_column(df: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...], title: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise RuntimeError(f"Колонка {explicit} не найдена для {title}")
        return explicit
    for name in candidates:
        if name in df.columns:
            return name
    raise RuntimeError(f"Не удалось определить колонку для {title}")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def add_bridge_features(df: pd.DataFrame, rp5_col: str, station_col: str) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["Date"].dt.month
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    out["rp5_x_sin"] = out[rp5_col] * out["sin_doy"]
    out["rp5_x_cos"] = out[rp5_col] * out["cos_doy"]

    station_dummies = pd.get_dummies(out[station_col].astype(str), prefix="station", drop_first=True)
    out = pd.concat([out, station_dummies], axis=1)
    return out


def build_design(df: pd.DataFrame, rp5_col: str) -> list[str]:
    base = [rp5_col, "sin_doy", "cos_doy", "rp5_x_sin", "rp5_x_cos"]
    station_terms = sorted([col for col in df.columns if col.startswith("station_")])
    return base + station_terms


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Калибровочный мост температуры rp5 -> Росгидромет")
    parser.add_argument("--input-csv", required=True, help="CSV с совпадающими датами и станциями rp5/Росгидромета")
    parser.add_argument("--adapter-json", default=None, help="JSON-маппинг колонок raw->canonical")
    parser.add_argument("--schema-only", action="store_true", help="Только валидация схемы и отчет без обучения")
    parser.add_argument("--strict-schema", action="store_true", help="Падать при невалидных датах/числах в целевых колонках")
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--station-col", default=None)
    parser.add_argument("--rp5-col", default=None)
    parser.add_argument("--hydromet-col", default=None)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-station-samples", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    return parser


def apply_adapter_mapping(df: pd.DataFrame, adapter_json: str | None) -> pd.DataFrame:
    if not adapter_json:
        return df
    mapping = json.loads(Path(adapter_json).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise RuntimeError("adapter-json должен содержать JSON-объект {raw_col: canonical_col}")
    rename_map: dict[str, str] = {}
    for raw_col, canonical_col in mapping.items():
        if not isinstance(raw_col, str) or not isinstance(canonical_col, str):
            raise RuntimeError("adapter-json должен содержать строковые пары {raw_col: canonical_col}")
        if raw_col in df.columns:
            rename_map[raw_col] = canonical_col
    return df.rename(columns=rename_map)


def validate_frame(
    df: pd.DataFrame,
    *,
    strict_schema: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    required_cols = ["Date", "station", "T_rp5", "T_hydromet"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Входная таблица не содержит обязательные колонки: {missing}")

    out = df.copy()
    rows_before = int(len(out))

    date_raw = out["Date"].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if strict_schema:
        bad_dates = int(out["Date"].isna().sum())
        if bad_dates > 0:
            raise RuntimeError(f"Невалидные даты в колонке Date: {bad_dates}")

    rp5_raw = out["T_rp5"].copy()
    hydromet_raw = out["T_hydromet"].copy()
    out["T_rp5"] = pd.to_numeric(out["T_rp5"], errors="coerce")
    out["T_hydromet"] = pd.to_numeric(out["T_hydromet"], errors="coerce")

    if strict_schema:
        bad_rp5 = int(rp5_raw.notna().sum() - out["T_rp5"].notna().sum())
        bad_hydromet = int(hydromet_raw.notna().sum() - out["T_hydromet"].notna().sum())
        if bad_rp5 > 0 or bad_hydromet > 0:
            raise RuntimeError(f"Невалидные числовые значения: bad_rp5={bad_rp5}, bad_hydromet={bad_hydromet}")

    out["station"] = out["station"].astype(str).str.strip()
    if strict_schema and (out["station"] == "").any():
        raise RuntimeError("Пустые station id после нормализации")

    out = out.dropna(subset=["Date", "station", "T_rp5", "T_hydromet"]).copy()
    report = {
        "rows_before_validation": rows_before,
        "rows_after_validation": int(len(out)),
        "rows_dropped": int(rows_before - len(out)),
        "stations_after_validation": int(out["station"].nunique()),
        "date_min": str(out["Date"].min().date()) if not out.empty else None,
        "date_max": str(out["Date"].max().date()) if not out.empty else None,
    }
    return out, report


def main() -> None:
    args = make_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    df.columns = [str(c).strip() for c in df.columns]
    df = apply_adapter_mapping(df, args.adapter_json)

    date_col = infer_column(df, args.date_col, DATE_CANDIDATES, "даты")
    station_col = infer_column(df, args.station_col, STATION_CANDIDATES, "станции")
    rp5_col = infer_column(df, args.rp5_col, RP5_TEMP_CANDIDATES, "температуры rp5")
    hydromet_col = infer_column(df, args.hydromet_col, HYDROMET_TEMP_CANDIDATES, "температуры Росгидромета")

    df = df.rename(columns={date_col: "Date", station_col: "station", rp5_col: "T_rp5", hydromet_col: "T_hydromet"}).copy()
    df, schema_report = validate_frame(df, strict_schema=args.strict_schema)
    if df.empty:
        raise RuntimeError("После валидации входная таблица пуста")
    df["year"] = df["Date"].dt.year

    station_counts = df["station"].value_counts()
    keep_stations = station_counts[station_counts >= args.min_station_samples].index
    df = df[df["station"].isin(keep_stations)].copy()

    df = add_bridge_features(df, rp5_col="T_rp5", station_col="station")
    features = build_design(df, rp5_col="T_rp5")

    train_mask = df["year"] <= args.train_end_year
    test_mask = (df["year"] >= args.test_start_year) & (df["year"] <= args.test_end_year)
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    if train.empty or test.empty:
        raise RuntimeError("После разбиения train/test одна из выборок пуста")

    model = Ridge(alpha=args.ridge_alpha)
    model.fit(train[features], train["T_hydromet"])

    train_pred = model.predict(train[features])
    test_pred = model.predict(test[features])
    df["T_hydromet_hat"] = model.predict(df[features])
    df["bridge_residual"] = df["T_hydromet_hat"] - df["T_hydromet"]

    baseline_train = compute_metrics(train["T_hydromet"], train["T_rp5"].to_numpy())
    baseline_test = compute_metrics(test["T_hydromet"], test["T_rp5"].to_numpy())
    bridge_train = compute_metrics(train["T_hydromet"], train_pred)
    bridge_test = compute_metrics(test["T_hydromet"], test_pred)

    outdir = args.output_dir
    if not outdir:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"outputs_runs/{ts}_rp5_hydromet_bridge"
    ensure_dir(outdir)
    save_json(Path(outdir) / "schema_report.json", schema_report)

    if args.schema_only:
        print(f"Schema validation OK, report saved: {Path(outdir) / 'schema_report.json'}")
        return

    coef_df = pd.DataFrame({"feature": features, "coef": model.coef_})
    coef_df.to_csv(Path(outdir) / "bridge_coefficients.csv", index=False)

    by_station_rows: list[dict[str, float | int | str]] = []
    for station, group in df.groupby("station"):
        by_station_rows.append(
            {
                "station": station,
                "n": int(len(group)),
                "baseline_mae": float(mean_absolute_error(group["T_hydromet"], group["T_rp5"])),
                "bridge_mae": float(mean_absolute_error(group["T_hydromet"], group["T_hydromet_hat"])),
                "baseline_bias": float((group["T_rp5"] - group["T_hydromet"]).mean()),
                "bridge_bias": float((group["T_hydromet_hat"] - group["T_hydromet"]).mean()),
            }
        )
    pd.DataFrame(by_station_rows).sort_values("bridge_mae").to_csv(Path(outdir) / "metrics_by_station.csv", index=False)

    monthly_rows: list[dict[str, float | int]] = []
    for month, group in df.groupby(df["Date"].dt.month):
        monthly_rows.append(
            {
                "month": int(month),
                "n": int(len(group)),
                "baseline_mae": float(mean_absolute_error(group["T_hydromet"], group["T_rp5"])),
                "bridge_mae": float(mean_absolute_error(group["T_hydromet"], group["T_hydromet_hat"])),
                "baseline_bias": float((group["T_rp5"] - group["T_hydromet"]).mean()),
                "bridge_bias": float((group["T_hydromet_hat"] - group["T_hydromet"]).mean()),
            }
        )
    pd.DataFrame(monthly_rows).sort_values("month").to_csv(Path(outdir) / "metrics_by_month.csv", index=False)

    df.to_csv(Path(outdir) / "bridge_predictions.csv", index=False)
    save_json(
        Path(outdir) / "metrics_summary.json",
        {
            "baseline_train": baseline_train,
            "baseline_test": baseline_test,
            "bridge_train": bridge_train,
            "bridge_test": bridge_test,
            "schema_report": schema_report,
            "ridge_alpha": args.ridge_alpha,
            "intercept": float(model.intercept_),
            "n_features": len(features),
        },
    )

    print(f"Saved bridge run: {outdir}")


if __name__ == "__main__":
    main()
