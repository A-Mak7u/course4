from __future__ import annotations

import argparse
import json
import netrc
from pathlib import Path

import pandas as pd

from appeears_point_task import login, submit_point_task
from volgograd_config import PROCESSED_ROOT, RAW_ROOT, VOLGOGRAD_SPEC, ensure_region_dirs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Разбить Volgograd MODIS AppEEARS на yearly point-tasks")
    parser.add_argument("--stations-csv", default=str(PROCESSED_ROOT / "volgograd_station_metadata_meteostat.csv"))
    parser.add_argument("--start-year", type=int, default=VOLGOGRAD_SPEC.start_year)
    parser.add_argument("--end-year", type=int, default=VOLGOGRAD_SPEC.end_year)
    parser.add_argument("--output-dir", default=str(RAW_ROOT / "modis_appeears_yearly"))
    parser.add_argument("--task-prefix", default=f"{VOLGOGRAD_SPEC.name}_mod11a1_point")
    parser.add_argument("--force", action="store_true")
    return parser


def load_earthdata_credentials() -> tuple[str, str]:
    auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise RuntimeError("В ~/.netrc нет записи для urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def resolve_years(start_year: int, end_year: int) -> list[int]:
    if end_year < start_year:
        raise RuntimeError("end_year не может быть меньше start_year")
    return list(range(start_year, end_year + 1))


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    stations = pd.read_csv(args.stations_csv)
    points = stations.rename(columns={"Cod": "id", "Y": "latitude", "X": "longitude"})[["id", "latitude", "longitude"]].copy()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_root / "points_used.csv", index=False)

    username, password = load_earthdata_credentials()
    token = login(username, password)

    submissions: list[dict] = []
    for year in resolve_years(args.start_year, args.end_year):
        year_dir = out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        submission_path = year_dir / "task_submission.json"

        if submission_path.exists() and not args.force:
            payload = json.loads(submission_path.read_text(encoding="utf-8"))
            submissions.append(payload)
            print(f"[yearly-submit] skip existing year={year} task_id={payload.get('task_id')}")
            continue

        task_name = f"{args.task_prefix}_{year}"
        task = submit_point_task(
            token=token,
            task_name=task_name,
            points_df=points,
            layers=[
                {"product": "MOD11A1.061", "layer": "LST_Day_1km"},
                {"product": "MOD11A1.061", "layer": "LST_Night_1km"},
            ],
            start_year=year,
            end_year=year,
        )
        submission_path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
        submissions.append(task)
        print(f"[yearly-submit] submitted year={year} task_id={task['task_id']}")

    manifest = {
        "years": resolve_years(args.start_year, args.end_year),
        "output_dir": str(out_root.resolve()),
        "points_csv": str((out_root / "points_used.csv").resolve()),
        "tasks": [
            {
                "year": year,
                "task_id": json.loads((out_root / str(year) / "task_submission.json").read_text(encoding="utf-8")).get("task_id"),
                "task_json": str((out_root / str(year) / "task_submission.json").resolve()),
            }
            for year in resolve_years(args.start_year, args.end_year)
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[yearly-submit] saved manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
