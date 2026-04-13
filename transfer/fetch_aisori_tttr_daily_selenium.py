#!/usr/bin/env python3
"""Automate AISORI-M daily TTTR export with headless Firefox.

This script is intended for long-running ingestion on a remote laptop node:
- logs into AISORI
- selects daily TTTR dataset
- selects stations (all by default)
- sets year range
- submits request and waits for completion
- downloads resulting file to a target directory

Credentials are read from CLI args or environment variables:
- AISORI_USER
- AISORI_PASS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait


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
        help="Target directory for downloaded files.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="tmp_rosgidromet_probe/aisori_snapshots",
        help="Directory for debug page snapshots on failure.",
    )
    parser.add_argument(
        "--dataset-label",
        default="TTTR - Температура и осадки",
        help="Visible option label for dataset selector.",
    )
    parser.add_argument(
        "--granularity-label",
        default="Сутки",
        help="Visible option label for granularity selector.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=4 * 60 * 60,
        help="Max wait time for server-side result generation.",
    )
    parser.add_argument(
        "--download-timeout-sec",
        type=int,
        default=60 * 60,
        help="Max wait time for browser file download completion.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Firefox with UI (default is headless).",
    )
    parser.add_argument(
        "--geckodriver",
        default=os.getenv("GECKODRIVER_PATH"),
        help="Path to geckodriver (optional).",
    )
    parser.add_argument(
        "--firefox-binary",
        default=os.getenv("FIREFOX_BINARY_PATH"),
        help="Path to Firefox binary (required on some snap-based installs).",
    )
    parser.add_argument(
        "--keep-browser",
        action="store_true",
        help="Do not close browser on success.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after filling conditions (before submit result).",
    )
    parser.add_argument(
        "--no-all-stations",
        action="store_true",
        help="Skip clicking 'all stations'.",
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        raise SystemExit("Username/password are required via args or AISORI_USER/AISORI_PASS env vars.")
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    return args


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def make_driver(
    download_dir: Path,
    geckodriver: Optional[str],
    firefox_binary: Optional[str],
    headless: bool,
) -> webdriver.Firefox:
    download_dir.mkdir(parents=True, exist_ok=True)
    options = FirefoxOptions()
    options.headless = headless
    if firefox_binary:
        options.binary_location = firefox_binary

    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.dir", str(download_dir.resolve()))
    profile.set_preference(
        "browser.helperApps.neverAsk.saveToDisk",
        ",".join(
            [
                "application/octet-stream",
                "application/zip",
                "application/x-zip-compressed",
                "text/plain",
                "text/csv",
                "application/csv",
                "application/vnd.ms-excel",
            ]
        ),
    )
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.alwaysOpenPanel", False)
    profile.set_preference("browser.download.useDownloadDir", True)
    profile.set_preference("browser.helperApps.alwaysAsk.force", False)
    profile.set_preference("pdfjs.disabled", True)
    options.profile = profile

    if geckodriver:
        service = FirefoxService(executable_path=geckodriver)
    else:
        service = FirefoxService()
    return webdriver.Firefox(service=service, options=options)


def wait_id(driver: webdriver.Firefox, element_id: str, timeout: int = 120):
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, element_id)))


def wait_clickable(driver: webdriver.Firefox, element_id: str, timeout: int = 120):
    return WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.ID, element_id)))


def clear_and_type(driver: webdriver.Firefox, element_id: str, value: str) -> None:
    elem = wait_id(driver, element_id)
    elem.clear()
    elem.send_keys(value)


def click(driver: webdriver.Firefox, element_id: str, timeout: int = 120) -> None:
    wait_clickable(driver, element_id, timeout).click()


def save_snapshot(driver: webdriver.Firefox, snapshot_dir: Path, label: str) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    html_path = snapshot_dir / f"{ts}_{label}.html"
    png_path = snapshot_dir / f"{ts}_{label}.png"
    html_path.write_text(driver.page_source, encoding="utf-8")
    driver.save_screenshot(str(png_path))
    log(f"Saved snapshot: {html_path}")
    log(f"Saved screenshot: {png_path}")


def parse_station_counter(text: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"\((\d+)\s*/\s*(\d+)\)", text or "")
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def wait_result_button_enabled(driver: webdriver.Firefox, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while True:
        btn = wait_id(driver, "j_idt60:butres1", timeout=120)
        disabled = btn.get_attribute("disabled")
        classes = btn.get_attribute("class") or ""
        enabled = disabled is None and "ui-state-disabled" not in classes
        if enabled:
            log("Result button is enabled.")
            return
        if time.time() > deadline:
            raise TimeoutException("Timed out waiting for result readiness.")
        time.sleep(5)


def newest_complete_file(download_dir: Path, after_ts: float) -> Optional[Path]:
    candidates: list[Path] = []
    for path in download_dir.glob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".part"):
            continue
        if path.stat().st_mtime < after_ts:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def wait_download(download_dir: Path, after_ts: float, timeout_sec: int) -> Path:
    deadline = time.time() + timeout_sec
    last_path: Optional[Path] = None
    last_size: Optional[int] = None
    stable_ticks = 0

    while True:
        latest = newest_complete_file(download_dir, after_ts)
        part_files = list(download_dir.glob("*.part"))
        if latest and not part_files:
            size = latest.stat().st_size
            if last_path == latest and last_size == size:
                stable_ticks += 1
            else:
                stable_ticks = 0
            last_path = latest
            last_size = size
            if stable_ticks >= 2:
                return latest
        if time.time() > deadline:
            raise TimeoutException("Timed out waiting for file download completion.")
        time.sleep(2)


def main() -> int:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()
    snapshot_dir = Path(args.snapshot_dir).expanduser()

    driver = make_driver(
        download_dir=download_dir,
        geckodriver=args.geckodriver,
        firefox_binary=args.firefox_binary,
        headless=not args.headful,
    )
    driver.set_page_load_timeout(180)
    wait = WebDriverWait(driver, 180)

    try:
        log("Opening login page.")
        driver.get(args.url)

        clear_and_type(driver, "j_idt13:usr", args.username)
        clear_and_type(driver, "j_idt13:pwd", args.password)
        click(driver, "j_idt13:j_idt27")
        wait.until(EC.presence_of_element_located((By.ID, "form1:newbut1")))
        log("Login succeeded.")

        click(driver, "form1:newbut1")
        wait.until(EC.presence_of_element_located((By.ID, "form1:istd")))
        log("Data selection page opened.")

        Select(wait_id(driver, "form1:razbd")).select_by_visible_text(args.granularity_label)
        Select(wait_id(driver, "form1:istd")).select_by_visible_text(args.dataset_label)
        log(f"Selected granularity={args.granularity_label!r}, dataset={args.dataset_label!r}.")

        if not args.no_all_stations:
            click(driver, "form1:p0stsel1")

            def all_stations_selected(_drv: webdriver.Firefox) -> bool:
                selinf = wait_id(_drv, "form1:selinf")
                selected, total = parse_station_counter(selinf.text)
                if selected is None or total is None:
                    return False
                return selected > 0 and selected == total

            WebDriverWait(driver, 180).until(all_stations_selected)
            selinf_text = wait_id(driver, "form1:selinf").text.strip()
            log(f"Stations selected: {selinf_text}")

        click(driver, "form1:okbut1")
        wait.until(EC.presence_of_element_located((By.ID, "form2:j_idt51")))
        log("Conditions page opened.")

        clear_and_type(driver, "form2:j_idt37:0:j_idt41", str(args.start_year))
        clear_and_type(driver, "form2:j_idt37:0:j_idt43", str(args.end_year))
        clear_and_type(driver, "form2:j_idt37:1:j_idt41", "")
        clear_and_type(driver, "form2:j_idt37:1:j_idt43", "")
        clear_and_type(driver, "form2:j_idt37:2:j_idt41", "")
        clear_and_type(driver, "form2:j_idt37:2:j_idt43", "")
        log(f"Year range set: {args.start_year}..{args.end_year}")

        click(driver, "form2:cb1")
        time.sleep(1.0)

        driver.execute_script(
            """
            const left = document.getElementById('form2:flist1_input');
            const right = document.getElementById('form2:flist2_input');
            if (left) {
              for (const o of left.options) o.selected = true;
            }
            if (right) {
              for (const o of right.options) o.selected = true;
            }
            """,
        )

        if args.dry_run:
            log("Dry-run requested; exiting before 'Result' submit.")
            return 0

        request_start_ts = time.time()
        click(driver, "form2:j_idt51")
        log("Result request submitted. Waiting for server-side preparation...")

        wait_result_button_enabled(driver, args.request_timeout_sec)

        click(driver, "j_idt60:butres1")
        log("Download requested. Waiting for file completion...")

        downloaded = wait_download(download_dir, after_ts=request_start_ts, timeout_sec=args.download_timeout_sec)
        result = {
            "downloaded_file": str(downloaded.resolve()),
            "size_bytes": downloaded.stat().st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(downloaded.stat().st_mtime)),
            "year_range": [args.start_year, args.end_year],
            "dataset": args.dataset_label,
            "granularity": args.granularity_label,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        log("AISORI export completed.")
        return 0

    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: {exc.__class__.__name__}: {exc}")
        save_snapshot(driver, snapshot_dir=snapshot_dir, label="error")
        return 2
    finally:
        if not args.keep_browser:
            driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
