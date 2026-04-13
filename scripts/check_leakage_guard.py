#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Finding:
    path: Path
    line: int
    code: str
    message: str
    snippet: str


K_FOLD_SHUFFLE_TRUE_RE = re.compile(r"\bKFold\s*\([^)]*\bshuffle\s*=\s*True\b")
EVAL_ON_TEST_RE = re.compile(
    r"evals\s*=\s*\[\s*\(\s*(?:dtest|d_test)\b|evals\s*=\s*\[\s*\([^)]*,\s*['\"]test['\"]",
    flags=re.IGNORECASE | re.DOTALL,
)


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        if K_FOLD_SHUFFLE_TRUE_RE.search(line):
            findings.append(
                Finding(
                    path=path,
                    line=i,
                    code="KFSHUF",
                    message="Найден KFold(..., shuffle=True). Для time-split используйте TimeSeriesSplit/хронологический split.",
                    snippet=line.strip(),
                )
            )

    # Guard against eval-on-test in XGBoost training.
    # We look for lines with evals= and inspect a short window to catch multiline calls.
    for i, line in enumerate(lines, start=1):
        if "evals" not in line.lower() or "=" not in line:
            continue
        window = "\n".join(lines[i - 1 : min(i + 5, len(lines))])
        if EVAL_ON_TEST_RE.search(window):
            findings.append(
                Finding(
                    path=path,
                    line=i,
                    code="EVALTEST",
                    message="Похоже на eval(...test...) в train. В тюнинге/early-stopping должен быть только внутренний val.",
                    snippet=line.strip(),
                )
            )

    return findings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guard на утечки: eval(...test...) и KFold(shuffle=True)."
    )
    p.add_argument(
        "--root",
        default=".",
        help="Корень репозитория.",
    )
    p.add_argument(
        "--globs",
        nargs="+",
        default=["xgb/**/*.py", "transfer/**/*.py"],
        help="Файловые маски для проверки.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    files: list[Path] = []
    for pattern in args.globs:
        files.extend(sorted(root.glob(pattern)))
    files = [p for p in files if p.is_file()]

    all_findings: list[Finding] = []
    for path in files:
        all_findings.extend(scan_file(path))

    if not all_findings:
        print("Leakage guard: OK")
        return 0

    print("Leakage guard: FAIL")
    for f in all_findings:
        rel = f.path.relative_to(root)
        print(f"{rel}:{f.line}: {f.code}: {f.message}")
        print(f"  >> {f.snippet}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
