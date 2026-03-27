from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


API_ROOT = "https://appeears.earthdatacloud.nasa.gov/api"


def login(username: str, password: str) -> str:
    response = requests.post(
        f"{API_ROOT}/login",
        auth=HTTPBasicAuth(username, password),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["token"]


def build_coordinates(points_df: pd.DataFrame) -> list[dict[str, str | float]]:
    coords: list[dict[str, str | float]] = []
    for _, row in points_df.iterrows():
        coords.append(
            {
                "id": str(row["id"]),
                "category": str(row.get("category", "station")),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        )
    return coords


def submit_point_task(
    *,
    token: str,
    task_name: str,
    points_df: pd.DataFrame,
    layers: list[dict[str, str]],
    start_year: int,
    end_year: int,
) -> dict:
    payload = {
        "task_type": "point",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": "01-01",
                    "endDate": "12-31",
                    "recurring": True,
                    "yearRange": [start_year, end_year],
                }
            ],
            "layers": layers,
            "coordinates": build_coordinates(points_df),
        },
    }
    response = requests.post(
        f"{API_ROOT}/task",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_task(token: str, task_id: str) -> dict:
    response = requests.get(
        f"{API_ROOT}/task/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def wait_for_task(token: str, task_id: str, poll_seconds: int = 30) -> dict:
    while True:
        task = get_task(token, task_id)
        status = str(task.get("status", "")).lower()
        if status in {"done", "complete", "completed"}:
            return task
        if status in {"error", "failed"}:
            raise RuntimeError(f"AppEEARS task {task_id} failed: {json.dumps(task, ensure_ascii=False)}")
        time.sleep(poll_seconds)


def list_bundle_files(token: str, task_id: str) -> list[dict]:
    response = requests.get(
        f"{API_ROOT}/bundle/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def download_bundle_files(
    *,
    token: str,
    task_id: str,
    bundle_files: Iterable[dict],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    headers = {"Authorization": f"Bearer {token}"}
    for item in bundle_files:
        file_id = item["file_id"]
        filename = item["file_name"]
        response = requests.get(
            f"{API_ROOT}/bundle/{task_id}/{file_id}",
            headers=headers,
            timeout=300,
        )
        response.raise_for_status()
        path = output_dir / filename
        path.write_bytes(response.content)
        saved.append(path)
    return saved


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Создание и выгрузка AppEEARS point-task")
    parser.add_argument("--points-csv", required=True, help="CSV с колонками id, latitude, longitude")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--product", required=True, help="Например MOD11A1.061")
    parser.add_argument("--layers", nargs="+", required=True, help="Например LST_Day_1km LST_Night_1km")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    points_df = pd.read_csv(args.points_csv)
    required = {"id", "latitude", "longitude"}
    missing = required.difference(points_df.columns)
    if missing:
        raise RuntimeError(f"В points-csv отсутствуют колонки: {sorted(missing)}")

    token = login(args.username, args.password)
    layers = [{"product": args.product, "layer": layer} for layer in args.layers]
    task = submit_point_task(
        token=token,
        task_name=args.task_name,
        points_df=points_df,
        layers=layers,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    task_id = task["task_id"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_submission.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Submitted task_id={task_id}")

    if args.submit_only:
        return

    final_task = wait_for_task(token, task_id, poll_seconds=args.poll_seconds)
    (output_dir / "task_final.json").write_text(json.dumps(final_task, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle_files = list_bundle_files(token, task_id)
    (output_dir / "bundle_files.json").write_text(json.dumps(bundle_files, indent=2, ensure_ascii=False), encoding="utf-8")
    saved = download_bundle_files(
        token=token,
        task_id=task_id,
        bundle_files=bundle_files,
        output_dir=output_dir,
    )
    print(f"Downloaded files: {len(saved)}")


if __name__ == "__main__":
    main()
