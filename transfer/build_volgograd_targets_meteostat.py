from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from meteostat import config as meteostat_config
from meteostat.api import daily
from meteostat.api.stations import Stations

from volgograd_config import PROCESSED_ROOT, VOLGOGRAD_SPEC, ensure_region_dirs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Сбор daily target T для Волгоградской области из Meteostat")
    parser.add_argument("--start-year", type=int, default=VOLGOGRAD_SPEC.start_year)
    parser.add_argument("--end-year", type=int, default=VOLGOGRAD_SPEC.end_year)
    parser.add_argument("--output-dir", default=str(PROCESSED_ROOT))
    return parser


def fetch_station_metadata() -> pd.DataFrame:
    stations = Stations()
    sql = """
    SELECT s.id AS station_id,
           n.name,
           s.country,
           s.region,
           s.latitude,
           s.longitude,
           s.elevation,
           s.timezone
    FROM stations s
    LEFT JOIN names n
      ON s.id = n.station AND n.language = 'en'
    WHERE s.country = :country
      AND s.region IN (:region_a, :region_b)
      AND s.latitude BETWEEN :south AND :north
      AND s.longitude BETWEEN :west AND :east
    ORDER BY s.latitude, s.longitude, s.id
    """
    meta = stations.query(
        sql,
        params={
            "country": VOLGOGRAD_SPEC.country,
            "region_a": VOLGOGRAD_SPEC.region_codes[0],
            "region_b": VOLGOGRAD_SPEC.region_codes[1],
            "south": VOLGOGRAD_SPEC.south,
            "north": VOLGOGRAD_SPEC.north,
            "west": VOLGOGRAD_SPEC.west,
            "east": VOLGOGRAD_SPEC.east,
        },
    ).copy()

    if meta.empty:
        raise RuntimeError("Не удалось найти станции Волгоградской области в Meteostat")

    meta["Cod"] = meta["station_id"].astype(int)
    meta = meta.rename(columns={"latitude": "Y", "longitude": "X"})
    return meta[["Cod", "station_id", "name", "country", "region", "X", "Y", "elevation", "timezone"]]


def fetch_daily_targets(meta: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    meteostat_config.block_large_requests = False
    station_ids = meta["station_id"].tolist()

    series = daily.daily(
        station_ids,
        start=date(start_year, 1, 1),
        end=date(end_year, 12, 31),
    )
    df = series.fetch().reset_index()
    if df.empty:
        raise RuntimeError("Meteostat daily вернул пустую таблицу")

    if "temp" not in df.columns:
        raise RuntimeError("В daily-таблице Meteostat нет колонки temp")

    out = df.rename(columns={"station": "station_id", "time": "Date", "temp": "T"}).copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.merge(meta[["Cod", "station_id", "X", "Y"]], on="station_id", how="left")
    out = out[["Cod", "station_id", "Date", "X", "Y", "T"]].sort_values(["Cod", "Date"]).reset_index(drop=True)
    return out


def build_coverage_table(targets: pd.DataFrame) -> pd.DataFrame:
    summary = (
        targets.assign(has_T=targets["T"].notna().astype(int))
        .groupby(["Cod", "station_id"], as_index=False)
        .agg(
            n_rows=("Date", "size"),
            n_T=("has_T", "sum"),
            T_missing_share=("has_T", lambda s: 1.0 - float(s.mean()) if len(s) else 1.0),
            date_min=("Date", "min"),
            date_max=("Date", "max"),
        )
        .sort_values(["n_T", "Cod"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = fetch_station_metadata()
    targets = fetch_daily_targets(meta, start_year=args.start_year, end_year=args.end_year)
    coverage = build_coverage_table(targets)

    meta.to_csv(outdir / "volgograd_station_metadata_meteostat.csv", index=False)
    targets.to_csv(outdir / f"volgograd_target_daily_meteostat_{args.start_year}_{args.end_year}.csv", index=False)
    coverage.to_csv(outdir / f"volgograd_target_coverage_meteostat_{args.start_year}_{args.end_year}.csv", index=False)

    print(f"Stations: {len(meta)}")
    print(f"Target rows: {len(targets)}")
    print(f"Output dir: {outdir}")


if __name__ == "__main__":
    main()
