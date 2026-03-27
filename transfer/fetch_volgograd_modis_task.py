from __future__ import annotations

import argparse
import json
import netrc
from pathlib import Path

from appeears_point_task import download_bundle_files, get_task, list_bundle_files, login, wait_for_task
from volgograd_config import RAW_ROOT, ensure_region_dirs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка и выгрузка уже созданной AppEEARS task для Волгограда")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--task-json", default=str(RAW_ROOT / "modis_appeears" / "task_submission.json"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default=str(RAW_ROOT / "modis_appeears"))
    parser.add_argument("--status-only", action="store_true")
    return parser


def load_earthdata_credentials() -> tuple[str, str]:
    auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise RuntimeError("В ~/.netrc нет записи для urs.earthdata.nasa.gov")
    username, _, password = auth
    return username, password


def resolve_task_id(task_id: str | None, task_json: str | Path) -> str:
    if task_id:
        return task_id
    payload = json.loads(Path(task_json).read_text(encoding="utf-8"))
    resolved = payload.get("task_id") or payload.get("taskid")
    if not resolved:
        raise RuntimeError(f"Не удалось определить task_id из {task_json}")
    return str(resolved)


def main() -> None:
    args = make_parser().parse_args()
    ensure_region_dirs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    username, password = load_earthdata_credentials()
    token = login(username, password)
    task_id = resolve_task_id(args.task_id, args.task_json)

    task = get_task(token, task_id)
    (output_dir / "task_status_latest.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Task {task_id} status={task.get('status')}")

    if args.status_only:
        return

    status = str(task.get("status", "")).lower()
    if status not in {"done", "complete", "completed"}:
        task = wait_for_task(token, task_id, poll_seconds=args.poll_seconds)
        (output_dir / "task_final.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle_files = list_bundle_files(token, task_id)
    (output_dir / "bundle_files.json").write_text(json.dumps(bundle_files, indent=2, ensure_ascii=False), encoding="utf-8")
    saved = download_bundle_files(token=token, task_id=task_id, bundle_files=bundle_files, output_dir=output_dir)
    print(f"Downloaded MODIS files: {len(saved)}")


if __name__ == "__main__":
    main()
