from __future__ import annotations

import argparse
import json
import netrc
import time
from pathlib import Path

import subprocess

from appeears_point_task import download_bundle_files, get_task, list_bundle_files, login
from volgograd_config import PROCESSED_ROOT, RAW_ROOT, ensure_region_dirs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка и выгрузка yearly Volgograd MODIS AppEEARS tasks")
    parser.add_argument("--input-dir", default=str(RAW_ROOT / "modis_appeears_yearly"))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--output-csv", default=str(PROCESSED_ROOT / "volgograd_modis_daily_2013_2023.csv"))
    parser.add_argument("--coverage-csv", default=str(PROCESSED_ROOT / "volgograd_modis_coverage_2013_2023.csv"))
    parser.add_argument("--status-only", action="store_true")
    return parser


def load_earthdata_credentials() -> tuple[str, str]:
    auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise RuntimeError("В ~/.netrc нет записи для urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def load_manifest(input_dir: Path) -> dict:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Не найден manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def year_has_results(year_dir: Path) -> bool:
    return any(year_dir.glob("*results.csv"))


def parse_combined_results(input_dir: Path, output_csv: Path, coverage_csv: Path) -> None:
    cmd = [
        str(Path.cwd() / ".venv_geo/bin/python"),
        "transfer/parse_volgograd_modis_appeears.py",
        "--input-dir",
        str(input_dir),
        "--points-csv",
        str(input_dir / "points_used.csv"),
        "--output-csv",
        str(output_csv),
        "--coverage-csv",
        str(coverage_csv),
    ]
    print(f"[yearly-fetch] run parser: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=Path.cwd(), check=True)


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    coverage_csv = Path(args.coverage_csv)
    manifest = load_manifest(input_dir)

    username, password = load_earthdata_credentials()
    token = login(username, password)

    while True:
        all_done = True
        for item in manifest["tasks"]:
            year = int(item["year"])
            task_id = str(item["task_id"])
            year_dir = input_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            task = get_task(token, task_id)
            (year_dir / "task_status_latest.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
            status = str(task.get("status", "")).lower()
            print(f"[yearly-fetch] year={year} task_id={task_id} status={status}", flush=True)

            if status not in {"done", "complete", "completed"}:
                all_done = False
                continue

            if year_has_results(year_dir):
                continue

            bundle_files = list_bundle_files(token, task_id)
            (year_dir / "bundle_files.json").write_text(json.dumps(bundle_files, indent=2, ensure_ascii=False), encoding="utf-8")
            saved = download_bundle_files(token=token, task_id=task_id, bundle_files=bundle_files, output_dir=year_dir)
            print(f"[yearly-fetch] downloaded year={year} files={len(saved)}", flush=True)

        if args.status_only:
            return

        if all_done and all(year_has_results(input_dir / str(item["year"])) for item in manifest["tasks"]):
            parse_combined_results(input_dir=input_dir, output_csv=output_csv, coverage_csv=coverage_csv)
            print(f"[yearly-fetch] saved combined csv: {output_csv}", flush=True)
            return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
