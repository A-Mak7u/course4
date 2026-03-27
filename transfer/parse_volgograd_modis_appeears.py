from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volgograd_config import PROCESSED_ROOT, RAW_ROOT

DATE_CANDIDATES = ("Date", "date", "datetime", "time")
STATION_CANDIDATES = ("id", "ID", "point_id", "Point", "point", "sample", "Sample", "subtask", "Subtask", "site", "Site", "name", "Name")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Парсинг AppEEARS point-results CSV в daily MODIS LST таблицу для Волгограда")
    parser.add_argument("--input-dir", default=str(RAW_ROOT / "modis_appeears"))
    parser.add_argument("--points-csv", default=str(RAW_ROOT / "modis_appeears" / "points_used.csv"))
    parser.add_argument("--output-csv", default=str(PROCESSED_ROOT / "volgograd_modis_daily_2013_2023.csv"))
    parser.add_argument("--coverage-csv", default=str(PROCESSED_ROOT / "volgograd_modis_coverage_2013_2023.csv"))
    return parser


def infer_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def normalize_station_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def choose_lst_column(columns: list[str], layer_name: str) -> str:
    exact = [col for col in columns if col == layer_name]
    if exact:
        return exact[0]
    suffix = [col for col in columns if col.endswith(layer_name)]
    if suffix:
        return sorted(suffix, key=len)[0]
    contains = [col for col in columns if layer_name in col]
    if contains:
        return sorted(contains, key=len)[0]
    raise RuntimeError(f"Не найдена колонка с AppEEARS layer {layer_name}")


def normalize_lst_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values <= 0)
    if values.dropna().empty:
        return values
    if values.dropna().quantile(0.95) > 2000:
        values = values * 0.02
    return values


def load_points(points_csv: str | Path) -> pd.DataFrame:
    points = pd.read_csv(points_csv)
    required = {"id", "latitude", "longitude"}
    missing = required.difference(points.columns)
    if missing:
        raise RuntimeError(f"В points-csv отсутствуют колонки: {sorted(missing)}")
    points["Cod"] = points["id"].map(normalize_station_id)
    points["Cod"] = pd.to_numeric(points["Cod"], errors="raise").astype("int64")
    return points[["Cod", "latitude", "longitude"]].drop_duplicates()


def parse_results_csv(path: Path, points: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["Cod", "Date", "LST_Day", "LST_Night"])

    date_col = infer_column(list(df.columns), DATE_CANDIDATES)
    if date_col is None:
        raise RuntimeError(f"Не удалось определить колонку даты в {path}")

    station_col = infer_column(list(df.columns), STATION_CANDIDATES)
    day_col = choose_lst_column(list(df.columns), "LST_Day_1km")
    night_col = choose_lst_column(list(df.columns), "LST_Night_1km")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    out["LST_Day"] = normalize_lst_series(df[day_col])
    out["LST_Night"] = normalize_lst_series(df[night_col])

    if station_col is not None:
        out["Cod"] = df[station_col].map(normalize_station_id)
    elif len(points) == 1:
        out["Cod"] = str(points.iloc[0]["Cod"])
    else:
        raise RuntimeError(f"Не удалось определить station id column в {path}")

    out["Cod"] = pd.to_numeric(out["Cod"], errors="raise").astype("int64")
    out = out[["Cod", "Date", "LST_Day", "LST_Night"]].copy()
    out = out.groupby(["Cod", "Date"], as_index=False).mean(numeric_only=True)
    return out


def main() -> None:
    args = make_parser().parse_args()
    input_dir = Path(args.input_dir)
    result_files = sorted(input_dir.rglob("*results.csv"))
    if not result_files:
        raise RuntimeError(f"В {input_dir} не найдено AppEEARS results CSV")

    points = load_points(args.points_csv)
    frames = [parse_results_csv(path, points) for path in result_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby(["Cod", "Date"], as_index=False).mean(numeric_only=True)
    combined = combined.sort_values(["Cod", "Date"]).reset_index(drop=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    coverage = combined.groupby("Cod").agg(
        modis_days_present=("Date", "nunique"),
        lst_day_present=("LST_Day", lambda s: int(s.notna().sum())),
        lst_night_present=("LST_Night", lambda s: int(s.notna().sum())),
    ).reset_index()
    coverage = coverage.merge(points[["Cod"]], on="Cod", how="right").fillna(0)
    for col in ["modis_days_present", "lst_day_present", "lst_night_present"]:
        coverage[col] = coverage[col].astype(int)
    coverage_path = Path(args.coverage_csv)
    coverage.to_csv(coverage_path, index=False)

    print(f"Parsed result files: {len(result_files)}")
    print(f"Rows: {len(combined)}")
    print(f"Stations: {combined['Cod'].nunique()}")
    print(f"Saved CSV: {output_path}")
    print(f"Saved coverage: {coverage_path}")


if __name__ == "__main__":
    main()
