#!/usr/bin/env python3
"""Automate AISORI-M daily TTTR export with Playwright (Firefox).

Workflow:
1. Login
2. Open "Выбор данных"
3. Select daily TTTR dataset
4. Select stations (all by default)
5. Open conditions page
6. Set date bounds
7. Submit result request and wait until download is ready
8. Save downloaded file
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


DEFAULT_URL = "http://aisori-m.meteo.ru/aisori-m/index0.xhtml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="AISORI login URL.")
    parser.add_argument("--username", default=os.getenv("AISORI_USER"))
    parser.add_argument("--password", default=os.getenv("AISORI_PASS"))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--download-dir",
        default="tmp_rosgidromet_probe/aisori_downloads",
        help="Directory where downloaded files are stored.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="tmp_rosgidromet_probe/aisori_playwright_snapshots",
        help="Directory for debug html/screenshots.",
    )
    parser.add_argument(
        "--dataset-label",
        default="TTTR - Температура и осадки",
        help="Visible label in dataset selector.",
    )
    parser.add_argument(
        "--granularity-label",
        default="Сутки",
        help="Visible label in granularity selector.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=4 * 60 * 60,
        help="Max wait for server-side result preparation.",
    )
    parser.add_argument(
        "--download-timeout-sec",
        type=int,
        default=60 * 60,
        help="Max wait for browser download event.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Launch browser with UI instead of headless mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop before clicking 'Результат'.",
    )
    parser.add_argument(
        "--no-all-stations",
        action="store_true",
        help="Skip selecting all stations.",
    )
    parser.add_argument(
        "--station-ids",
        default=None,
        help="Comma-separated WMO station ids to select explicitly (e.g. 34560,34476).",
    )
    parser.add_argument(
        "--station-ids-file",
        default=None,
        help="Path to text file with station ids (one id per line).",
    )
    parser.add_argument(
        "--allow-partial-stations",
        action="store_true",
        help="Do not fail if only part of requested station ids is available on the page.",
    )
    parser.add_argument(
        "--month-from",
        type=int,
        default=None,
        help="Optional month lower bound (1..12).",
    )
    parser.add_argument(
        "--month-to",
        type=int,
        default=None,
        help="Optional month upper bound (1..12).",
    )
    parser.add_argument(
        "--day-from",
        type=int,
        default=None,
        help="Optional day lower bound (1..31).",
    )
    parser.add_argument(
        "--day-to",
        type=int,
        default=None,
        help="Optional day upper bound (1..31).",
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        raise SystemExit("Username/password are required via args or AISORI_USER/AISORI_PASS.")
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    return args


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def save_debug(page, snapshot_dir: Path, label: str) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    html_path = snapshot_dir / f"{ts}_{label}.html"
    png_path = snapshot_dir / f"{ts}_{label}.png"
    html_path.write_text(page.content(), encoding="utf-8")
    page.screenshot(path=str(png_path), full_page=True)
    log(f"Saved debug html: {html_path}")
    log(f"Saved screenshot: {png_path}")


def parse_selinf(text: str) -> tuple[int | None, int | None]:
    m = re.search(r"\((\d+)\s*/\s*(\d+)\)", text or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def fill_optional(page, selector: str, value: int | None) -> None:
    page.fill(selector, "" if value is None else str(value))


def fill_range_row(page, row_label: str, from_value: int | None, to_value: int | None) -> None:
    row = page.locator("#form2\\:j_idt37_data tr", has_text=row_label).first
    if row.count() == 0:
        raise RuntimeError(f"Range row with label {row_label!r} not found.")
    inputs = row.locator("input")
    if inputs.count() < 2:
        raise RuntimeError(f"Range row {row_label!r} does not contain two input fields.")
    inputs.nth(0).fill("" if from_value is None else str(from_value))
    inputs.nth(1).fill("" if to_value is None else str(to_value))


def wait_download_ready(page, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while True:
        butres = page.locator("#j_idt60\\:butres1")
        butclose = page.locator("#j_idt60\\:but1close")
        if butres.count() > 0:
            disabled_res = butres.get_attribute("disabled")
            disabled_close = butclose.get_attribute("disabled") if butclose.count() else "disabled"
            if disabled_res is None:
                return
            if disabled_close is None and disabled_res is not None:
                raise RuntimeError("AISORI request finished with error (close enabled, result still disabled).")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for result readiness.")
        time.sleep(2)


def parse_station_ids(raw: str | None, file_path: str | None) -> list[str]:
    ids: list[str] = []
    if raw:
        for chunk in raw.split(","):
            token = chunk.strip()
            if token:
                ids.append(token)
    if file_path:
        text = Path(file_path).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0])
    normalized: list[str] = []
    seen: set[str] = set()
    for sid in ids:
        sid2 = "".join(ch for ch in sid if ch.isdigit())
        if not sid2:
            continue
        if sid2 not in seen:
            seen.add(sid2)
            normalized.append(sid2)
    return normalized


def get_station_value_map(page) -> dict[str, str]:
    option_values: list[str] = page.eval_on_selector_all(
        "#form1\\:hlist1_input option",
        "els => els.map(e => e.value)",
    )
    out: dict[str, str] = {}
    for value in option_values:
        m = re.match(r"^\s*(\d{5,6})\b", value or "")
        if not m:
            continue
        sid = m.group(1)
        if sid not in out:
            out[sid] = value
    return out


def select_specific_stations(page, station_ids: Iterable[str], allow_partial: bool) -> tuple[list[str], list[str]]:
    value_map = get_station_value_map(page)
    requested = [str(x).strip() for x in station_ids if str(x).strip()]
    available = sorted(value_map.keys(), key=int)
    selected_ids = [sid for sid in requested if sid in value_map]
    missing_ids = [sid for sid in requested if sid not in value_map]
    if not selected_ids:
        raise RuntimeError(
            "Не удалось выбрать станции: ни один station id не найден в текущем списке AISORI. "
            f"requested={requested}, available_count={len(available)}"
        )
    if missing_ids and not allow_partial:
        raise RuntimeError(
            "Не все requested station ids доступны в AISORI: "
            f"missing={missing_ids}; requested={requested}; available_count={len(available)}"
        )

    values = [value_map[sid] for sid in selected_ids]
    page.select_option("#form1\\:hlist1_input", value=values)
    page.click("#form1\\:p1stsel1")

    deadline = time.time() + 180
    need = len(selected_ids)
    while True:
        text = page.locator("#form1\\:selinf").inner_text()
        selected_count, _ = parse_selinf(text)
        if selected_count is not None and selected_count >= need:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting selected stations counter to reach >= {need}. Last selinf={text!r}"
            )
        time.sleep(0.5)
    return selected_ids, missing_ids


def main() -> int:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()
    snapshot_dir = Path(args.snapshot_dir).expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)
    requested_station_ids = parse_station_ids(args.station_ids, args.station_ids_file)
    selected_station_ids: list[str] = []
    missing_station_ids: list[str] = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=not args.headful)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        try:
            log("Opening login page.")
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)

            page.fill("#j_idt13\\:usr", args.username)
            page.fill("#j_idt13\\:pwd", args.password)
            page.click("#j_idt13\\:j_idt27")
            page.wait_for_selector("#form1\\:newbut1", timeout=120_000)
            log("Login succeeded.")

            page.click("#form1\\:newbut1")
            page.wait_for_selector("#form1\\:istd", timeout=120_000)
            log("Selection page opened.")

            page.select_option("#form1\\:razbd", label=args.granularity_label)
            page.select_option("#form1\\:istd", label=args.dataset_label)
            log(f"Selected granularity={args.granularity_label!r}, dataset={args.dataset_label!r}.")

            if requested_station_ids:
                selected_station_ids, missing_station_ids = select_specific_stations(
                    page,
                    requested_station_ids,
                    allow_partial=args.allow_partial_stations,
                )
                text = page.locator("#form1\\:selinf").inner_text()
                log(
                    "Selected explicit station ids: "
                    f"{len(selected_station_ids)} / requested={len(requested_station_ids)}; selinf={text.strip()}"
                )
                if missing_station_ids:
                    log(f"Missing station ids on AISORI page: {missing_station_ids}")
            elif not args.no_all_stations:
                page.click("#form1\\:p0stsel1")
                deadline = time.time() + 180
                while True:
                    text = page.locator("#form1\\:selinf").inner_text()
                    selected, total = parse_selinf(text)
                    if selected is not None and total is not None and selected > 0 and selected == total:
                        log(f"Stations selected: {text.strip()}")
                        break
                    if time.time() > deadline:
                        raise TimeoutError(f"Timed out waiting all stations selection. Last selinf={text!r}")
                    time.sleep(1)

            page.click("#form1\\:okbut1")
            page.wait_for_selector("#form2\\:j_idt51", timeout=120_000)
            log("Conditions page opened.")

            fill_range_row(page, "Год", args.start_year, args.end_year)
            fill_range_row(page, "Месяц", args.month_from, args.month_to)
            fill_range_row(page, "День", args.day_from, args.day_to)
            log(f"Year range set to {args.start_year}..{args.end_year}.")

            # Move all available fields into request fields (includes temperature/precip columns).
            before_count = page.locator("#form2\\:flist2_input option").count()
            page.click("#form2\\:cb1")
            deadline = time.time() + 20
            while time.time() < deadline:
                after_count = page.locator("#form2\\:flist2_input option").count()
                if after_count >= before_count:
                    if after_count > before_count:
                        log(f"Request fields expanded: {before_count} -> {after_count}")
                    break
                time.sleep(0.2)

            if args.dry_run:
                log("Dry-run completed (before result submit).")
                return 0

            page.click("#form2\\:j_idt51")
            log("Result request submitted; waiting for readiness...")

            wait_download_ready(page, timeout_sec=args.request_timeout_sec)
            log("Result is ready; starting download.")

            # Step 1: move from waiting dialog to result page.
            page.click("#j_idt60\\:butres1")
            page.wait_for_selector("#form3\\:j_idt36", timeout=120_000)

            # Step 2: actual file download from result page.
            with page.expect_download(timeout=args.download_timeout_sec * 1000) as download_info:
                page.click("#form3\\:j_idt36")
            download = download_info.value

            suggested = download.suggested_filename or f"aisori_{int(time.time())}.dat"
            target = download_dir / suggested
            download.save_as(str(target))

            result = {
                "downloaded_file": str(target.resolve()),
                "size_bytes": target.stat().st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(target.stat().st_mtime)),
                "dataset": args.dataset_label,
                "granularity": args.granularity_label,
                "years": [args.start_year, args.end_year],
                "requested_station_ids": requested_station_ids,
                "selected_station_ids": selected_station_ids,
                "missing_station_ids": missing_station_ids,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            log("AISORI export finished successfully.")
            return 0
        except (PlaywrightTimeoutError, TimeoutError, RuntimeError) as exc:
            log(f"ERROR: {exc.__class__.__name__}: {exc}")
            save_debug(page, snapshot_dir=snapshot_dir, label="error")
            return 2
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
