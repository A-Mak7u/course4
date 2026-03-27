from __future__ import annotations

import argparse
import calendar
import contextlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import cdsapi
import pandas as pd
import xarray as xr

from volgograd_config import INTERIM_ROOT, PROCESSED_ROOT, RAW_ROOT, VOLGOGRAD_SPEC, ensure_region_dirs

ERA5_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "evaporation",
)

ERA5_RENAME_MAP = {
    "t2m": "Temperature_2m",
    "d2m": "Dewpoint_2m",
    "sp": "Surface_pressure",
    "tp": "Total_precipitation",
    "e": "Evaporation",
}

ERA5_POINT_COLUMNS = ["t2m", "d2m", "sp", "tp", "e"]


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            tmp_file.write(text)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def is_valid_archive(path: Path) -> bool:
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return bool(zf.namelist())
    except zipfile.BadZipFile:
        return False


def is_valid_month_csv(path: Path, year: int, month: int, station_count: int) -> bool:
    if not path.exists():
        return False
    required = {"Cod", "Date", *ERA5_RENAME_MAP.values(), "X_final", "Y_final"}
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return False
    if required.difference(df.columns):
        return False
    expected_days = calendar.monthrange(year, month)[1]
    expected_rows = expected_days * station_count
    if len(df) != expected_rows:
        return False
    if df["Cod"].nunique() != station_count:
        return False
    dates = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.normalize()
    if dates.isna().any():
        return False
    if dates.nunique() != expected_days:
        return False
    return True


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Сборка суточных ERA5-признаков по станциям Волгоградской области")
    parser.add_argument("--stations-csv", default=str(PROCESSED_ROOT / "volgograd_station_metadata_meteostat.csv"))
    parser.add_argument("--start-year", type=int, default=VOLGOGRAD_SPEC.start_year)
    parser.add_argument("--end-year", type=int, default=VOLGOGRAD_SPEC.end_year)
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Явный список лет вместо диапазона")
    parser.add_argument("--months", nargs="+", type=int, default=None, help="Явный список месяцев 1..12")
    parser.add_argument("--raw-dir", default=str(RAW_ROOT / "era5"))
    parser.add_argument("--interim-dir", default=str(INTERIM_ROOT / "era5_daily_yearly"))
    parser.add_argument("--processed-dir", default=str(PROCESSED_ROOT))
    parser.add_argument("--bbox-pad", type=float, default=0.25, help="Поля вокруг bbox для интерполяции на краях")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-process", action="store_true")
    return parser


def resolve_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        return sorted(set(args.years))
    return list(range(args.start_year, args.end_year + 1))


def resolve_periods(years: list[int], months: list[int] | None = None) -> list[tuple[int, int]]:
    resolved_months = sorted(set(months)) if months else list(range(1, 13))
    invalid = [month for month in resolved_months if month < 1 or month > 12]
    if invalid:
        raise RuntimeError(f"Некорректные номера месяцев: {invalid}")
    return [(year, month) for year in years for month in resolved_months]


def load_stations(path: str | Path) -> pd.DataFrame:
    stations = pd.read_csv(path)
    required = {"Cod", "X", "Y"}
    missing = required.difference(stations.columns)
    if missing:
        raise RuntimeError(f"В stations-csv отсутствуют колонки: {sorted(missing)}")
    stations = stations[["Cod", "X", "Y"]].dropna().copy()
    stations["Cod"] = pd.to_numeric(stations["Cod"], errors="raise").astype("int64")
    stations["X"] = stations["X"].astype(float)
    stations["Y"] = stations["Y"].astype(float)
    stations = stations.drop_duplicates(subset=["Cod"]).sort_values("Cod").reset_index(drop=True)
    return stations


def build_request(year: int, month: int, bbox_pad: float) -> dict[str, list[str] | str]:
    north = VOLGOGRAD_SPEC.north + bbox_pad
    west = VOLGOGRAD_SPEC.west - bbox_pad
    south = VOLGOGRAD_SPEC.south - bbox_pad
    east = VOLGOGRAD_SPEC.east + bbox_pad
    last_day = calendar.monthrange(year, month)[1]
    return {
        "product_type": ["reanalysis"],
        "variable": list(ERA5_VARIABLES),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, last_day + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
    }


def month_archive_path(raw_dir: Path, year: int, month: int) -> Path:
    return raw_dir / f"volgograd_era5_hourly_{year}_{month:02d}.zip"


def month_csv_path(interim_dir: Path, year: int, month: int) -> Path:
    return interim_dir / f"volgograd_era5_daily_{year}_{month:02d}.csv"


def download_month_archive(
    client: cdsapi.Client,
    year: int,
    month: int,
    raw_dir: Path,
    bbox_pad: float,
    force: bool = False,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = month_archive_path(raw_dir, year, month)
    if archive_path.exists() and not force:
        if is_valid_archive(archive_path):
            print(f"[ERA5] period={year}-{month:02d} download skipped, archive already exists: {archive_path}")
            return archive_path
        print(f"[ERA5] period={year}-{month:02d} invalid archive detected, redownloading: {archive_path}")
        archive_path.unlink()

    request = build_request(year, month, bbox_pad=bbox_pad)
    print(f"[ERA5] period={year}-{month:02d} download started")
    result = client.retrieve("reanalysis-era5-single-levels", request)
    tmp_archive = archive_path.with_suffix(f"{archive_path.suffix}.tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()
    result.download(str(tmp_archive))
    tmp_archive.replace(archive_path)
    print(f"[ERA5] period={year}-{month:02d} download finished: {archive_path}")
    return archive_path


@contextlib.contextmanager
def open_era5_zip_dataset(zip_path: Path) -> Iterator[xr.Dataset]:
    opened_parts: list[xr.Dataset] = []
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for member in zf.namelist():
            member_path = tmpdir / member
            member_path.parent.mkdir(parents=True, exist_ok=True)
            member_path.write_bytes(zf.read(member))
            opened_parts.append(xr.open_dataset(member_path, engine="netcdf4"))
        merged = xr.merge(opened_parts, compat="override")
        try:
            yield merged
        finally:
            merged.close()
            for ds in opened_parts:
                ds.close()


def process_year_archive(zip_path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    with open_era5_zip_dataset(zip_path) as ds:
        missing_vars = [name for name in ERA5_POINT_COLUMNS if name not in ds.data_vars]
        if missing_vars:
            raise RuntimeError(f"В {zip_path} отсутствуют ERA5-переменные: {missing_vars}")

        lon = xr.DataArray(stations["X"].to_numpy(), dims="station")
        lat = xr.DataArray(stations["Y"].to_numpy(), dims="station")
        interp = ds[list(ERA5_POINT_COLUMNS)].interp(longitude=lon, latitude=lat)
        interp = interp.assign_coords(station=("station", stations["Cod"].to_numpy()))

        frame = interp.to_dataframe().reset_index()
        frame["Date"] = pd.to_datetime(frame["valid_time"], utc=True).dt.normalize()
        daily = frame.groupby(["station", "Date"], as_index=False)[list(ERA5_POINT_COLUMNS)].mean()
        daily = daily.rename(columns={"station": "Cod", **ERA5_RENAME_MAP})
        daily["Cod"] = daily["Cod"].astype("int64")

        station_meta = stations.rename(columns={"X": "X_final", "Y": "Y_final"})
        daily = daily.merge(station_meta, on="Cod", how="left", validate="many_to_one")
        daily = daily[["Cod", "Date", *ERA5_RENAME_MAP.values(), "X_final", "Y_final"]]
        daily = daily.sort_values(["Cod", "Date"]).reset_index(drop=True)
        return daily


def build_combined_outputs(
    *,
    interim_dir: Path,
    processed_dir: Path,
    periods: list[tuple[int, int]],
    stations: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year, month in periods:
        period_path = month_csv_path(interim_dir, year, month)
        if not period_path.exists():
            raise RuntimeError(f"Не найден промежуточный ERA5 CSV для period={year}-{month:02d}: {period_path}")
        frames.append(pd.read_csv(period_path, parse_dates=["Date"]))

    combined = pd.concat(frames, ignore_index=True).sort_values(["Cod", "Date"]).reset_index(drop=True)
    if combined["Date"].dt.tz is None:
        combined["Date"] = combined["Date"].dt.tz_localize("UTC")

    combined_path = processed_dir / f"volgograd_era5_daily_{start_year}_{end_year}.csv"
    atomic_write_csv(combined, combined_path)

    expected_days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D", tz="UTC")
    coverage = (
        combined.groupby("Cod")["Date"].nunique().rename("era5_days_present").reset_index()
        .merge(stations[["Cod"]], on="Cod", how="right")
        .fillna({"era5_days_present": 0})
    )
    coverage["era5_days_present"] = coverage["era5_days_present"].astype(int)
    coverage["expected_days"] = len(expected_days)
    coverage["coverage_ratio"] = coverage["era5_days_present"] / coverage["expected_days"]
    coverage_path = processed_dir / f"volgograd_era5_coverage_{start_year}_{end_year}.csv"
    atomic_write_csv(coverage, coverage_path)
    return combined_path, coverage_path


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    if args.download_only and args.process_only:
        raise RuntimeError("Нельзя одновременно задать --download-only и --process-only")

    stations = load_stations(args.stations_csv)
    years = resolve_years(args)
    periods = resolve_periods(years, months=args.months)
    raw_dir = Path(args.raw_dir)
    interim_dir = Path(args.interim_dir)
    processed_dir = Path(args.processed_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    station_count = len(stations)

    client = None if args.process_only else cdsapi.Client(quiet=False, progress=False)

    for year, month in periods:
        archive_path = month_archive_path(raw_dir, year, month)
        if not args.process_only:
            archive_path = download_month_archive(
                client,
                year,
                month,
                raw_dir,
                bbox_pad=args.bbox_pad,
                force=args.force_download,
            )
        if args.download_only:
            continue

        out_csv = month_csv_path(interim_dir, year, month)
        if out_csv.exists() and not args.force_process:
            if is_valid_month_csv(out_csv, year, month, station_count):
                print(f"[ERA5] period={year}-{month:02d} process skipped, CSV already exists: {out_csv}")
                continue
            print(f"[ERA5] period={year}-{month:02d} invalid CSV detected, rebuilding: {out_csv}")
            out_csv.unlink()

        if not archive_path.exists():
            raise RuntimeError(f"Для period={year}-{month:02d} не найден ERA5 archive: {archive_path}")

        print(f"[ERA5] period={year}-{month:02d} processing started")
        daily = process_year_archive(archive_path, stations)
        atomic_write_csv(daily, out_csv)
        print(f"[ERA5] period={year}-{month:02d} processing finished: rows={len(daily)} -> {out_csv}")

    if not args.download_only:
        combined_path, coverage_path = build_combined_outputs(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            periods=periods,
            stations=stations,
            start_year=min(years),
            end_year=max(years),
        )
        meta = {
            "stations_csv": str(Path(args.stations_csv).resolve()),
            "years": years,
            "periods": [f"{year}-{month:02d}" for year, month in periods],
            "raw_dir": str(raw_dir.resolve()),
            "interim_dir": str(interim_dir.resolve()),
            "processed_combined_csv": str(combined_path.resolve()),
            "coverage_csv": str(coverage_path.resolve()),
            "method": "ERA5 hourly regional monthly bbox -> xarray interp -> daily mean by station/date",
            "variables": list(ERA5_VARIABLES),
        }
        meta_path = processed_dir / f"volgograd_era5_build_meta_{min(years)}_{max(years)}.json"
        atomic_write_text(json.dumps(meta, indent=2, ensure_ascii=False), meta_path)
        print(f"[ERA5] combined CSV: {combined_path}")
        print(f"[ERA5] coverage CSV: {coverage_path}")
        print(f"[ERA5] meta JSON: {meta_path}")


if __name__ == "__main__":
    main()
