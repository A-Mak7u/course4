from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volgograd_config import PROCESSED_ROOT

FINAL_COLUMNS = [
    "Cod",
    "Date",
    "T",
    "Temperature_2m",
    "Dewpoint_2m",
    "Surface_pressure",
    "Total_precipitation",
    "Evaporation",
    "LST_Day",
    "LST_Night",
    "X_final",
    "Y_final",
]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Склейка финального волгоградского датасета под XGB transfer")
    parser.add_argument("--target-csv", default=str(PROCESSED_ROOT / "volgograd_target_daily_meteostat_2013_2023.csv"))
    parser.add_argument("--era5-csv", default=str(PROCESSED_ROOT / "volgograd_era5_daily_2013_2023.csv"))
    parser.add_argument("--modis-csv", default=str(PROCESSED_ROOT / "volgograd_modis_daily_2013_2023.csv"))
    parser.add_argument("--output-csv", default=str(PROCESSED_ROOT / "volgograd_final_2013_2023_T_ERA5_LST_daynight.csv"))
    parser.add_argument("--allow-missing-era5", action="store_true")
    parser.add_argument("--allow-missing-modis", action="store_true")
    return parser


def normalize_daily_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=True).dt.normalize()
    return out


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def load_target(path: str | Path) -> pd.DataFrame:
    target = pd.read_csv(path)
    required = {"Cod", "Date", "T", "X", "Y"}
    missing = required.difference(target.columns)
    if missing:
        raise RuntimeError(f"В target-csv отсутствуют колонки: {sorted(missing)}")
    target = normalize_daily_dates(target)
    target["Cod"] = pd.to_numeric(target["Cod"], errors="raise").astype("int64")
    target = target.rename(columns={"X": "X_final", "Y": "Y_final"})
    return target[["Cod", "Date", "T", "X_final", "Y_final"]].copy()


def load_era5(path: str | Path, allow_missing: bool) -> pd.DataFrame:
    if not Path(path).exists():
        if allow_missing:
            return pd.DataFrame(columns=["Cod", "Date", "Temperature_2m", "Dewpoint_2m", "Surface_pressure", "Total_precipitation", "Evaporation"])
        raise RuntimeError(f"Не найден ERA5 CSV: {path}")
    era5 = pd.read_csv(path)
    required = {"Cod", "Date", "Temperature_2m", "Dewpoint_2m", "Surface_pressure", "Total_precipitation", "Evaporation"}
    missing = required.difference(era5.columns)
    if missing:
        raise RuntimeError(f"В era5-csv отсутствуют колонки: {sorted(missing)}")
    era5 = normalize_daily_dates(era5)
    era5["Cod"] = pd.to_numeric(era5["Cod"], errors="raise").astype("int64")
    return era5[list(required)].copy()


def load_modis(path: str | Path, allow_missing: bool) -> pd.DataFrame:
    if not Path(path).exists():
        if allow_missing:
            return pd.DataFrame(columns=["Cod", "Date", "LST_Day", "LST_Night"])
        raise RuntimeError(f"Не найден MODIS CSV: {path}")
    modis = pd.read_csv(path)
    required = {"Cod", "Date", "LST_Day", "LST_Night"}
    missing = required.difference(modis.columns)
    if missing:
        raise RuntimeError(f"В modis-csv отсутствуют колонки: {sorted(missing)}")
    modis = normalize_daily_dates(modis)
    modis["Cod"] = pd.to_numeric(modis["Cod"], errors="raise").astype("int64")
    return modis[list(required)].copy()


def main() -> None:
    args = make_parser().parse_args()

    target = load_target(args.target_csv)
    era5 = load_era5(args.era5_csv, allow_missing=args.allow_missing_era5)
    modis = load_modis(args.modis_csv, allow_missing=args.allow_missing_modis)

    merged = target.merge(era5, on=["Cod", "Date"], how="left", validate="one_to_one")
    merged = merged.merge(modis, on=["Cod", "Date"], how="left", validate="one_to_one")
    merged = ensure_columns(merged, FINAL_COLUMNS)
    merged = merged[FINAL_COLUMNS].sort_values(["Cod", "Date"]).reset_index(drop=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Final rows: {len(merged)}")
    print(f"Stations: {merged['Cod'].nunique()}")
    print(f"Date range: {merged['Date'].min()} .. {merged['Date'].max()}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
