from __future__ import annotations

import argparse
import json
import netrc
from pathlib import Path

import pandas as pd

from appeears_point_task import download_bundle_files, list_bundle_files, login, submit_point_task, wait_for_task
from volgograd_config import PROCESSED_ROOT, RAW_ROOT, VOLGOGRAD_SPEC, ensure_region_dirs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AppEEARS point-task для MODIS LST по станциям Волгоградской области")
    parser.add_argument("--stations-csv", default=str(PROCESSED_ROOT / "volgograd_station_metadata_meteostat.csv"))
    parser.add_argument("--task-name", default=f"{VOLGOGRAD_SPEC.name}_mod11a1_point_2013_2023")
    parser.add_argument("--start-year", type=int, default=VOLGOGRAD_SPEC.start_year)
    parser.add_argument("--end-year", type=int, default=VOLGOGRAD_SPEC.end_year)
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default=str(RAW_ROOT / "modis_appeears"))
    return parser


def load_earthdata_credentials() -> tuple[str, str]:
    auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise RuntimeError("В ~/.netrc нет записи для urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    stations = pd.read_csv(args.stations_csv)
    points = stations.rename(columns={"Cod": "id", "Y": "latitude", "X": "longitude"})[["id", "latitude", "longitude"]].copy()

    username, password = load_earthdata_credentials()
    token = login(username, password)
    task = submit_point_task(
        token=token,
        task_name=args.task_name,
        points_df=points,
        layers=[
            {"product": "MOD11A1.061", "layer": "LST_Day_1km"},
            {"product": "MOD11A1.061", "layer": "LST_Night_1km"},
        ],
        start_year=args.start_year,
        end_year=args.end_year,
    )
    task_id = task["task_id"]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "task_submission.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    points.to_csv(outdir / "points_used.csv", index=False)
    print(f"Submitted MODIS AppEEARS task_id={task_id}")

    if args.submit_only:
        return

    final_task = wait_for_task(token, task_id, poll_seconds=args.poll_seconds)
    (outdir / "task_final.json").write_text(json.dumps(final_task, indent=2, ensure_ascii=False), encoding="utf-8")
    bundle_files = list_bundle_files(token, task_id)
    (outdir / "bundle_files.json").write_text(json.dumps(bundle_files, indent=2, ensure_ascii=False), encoding="utf-8")
    saved = download_bundle_files(token=token, task_id=task_id, bundle_files=bundle_files, output_dir=outdir)
    print(f"Downloaded MODIS files: {len(saved)}")


if __name__ == "__main__":
    main()
