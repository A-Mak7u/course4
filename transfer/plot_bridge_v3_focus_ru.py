#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "baseline": "Базовая линия RP5",
    "xgb_delta_gated": "XGB delta + gate",
    "xgb_delta_gated_adaptive": "XGB delta + адаптивный gate",
    "xgb_delta_gated_adaptive_safeguard": "XGB delta + адаптивный gate + safeguard",
    "xgb_delta_clustered_v3_gated": "Кластерный v3 + gate",
    "xgb_delta_selector_station": "Meta-selector по станциям",
    "xgb_delta_selector_station_month": "Meta-selector station+month",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Фокусный график по bridge v3 (русские подписи).")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Опциональный путь для сохранения PNG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    vm = pd.read_csv(run_dir / "variant_metrics.csv")
    risk = pd.read_csv(run_dir / "station_risk_summary_test.csv")

    base_order = [
        "baseline",
        "xgb_delta_gated",
        "xgb_delta_gated_adaptive",
        "xgb_delta_gated_adaptive_safeguard",
        "xgb_delta_clustered_v3_gated",
        "xgb_delta_selector_station",
        "xgb_delta_selector_station_month",
    ]
    ordered_variants = [v for v in base_order if v in vm["variant"].unique().tolist()]
    test = vm[(vm["split"] == "test") & (vm["variant"].isin(ordered_variants))].copy()
    test["variant"] = pd.Categorical(test["variant"], categories=ordered_variants, ordered=True)
    test = test.sort_values("variant")

    risk_variants = [v for v in ordered_variants if v != "baseline"]
    risk = risk[risk["variant"].isin(risk_variants)].copy()
    risk = risk.set_index("variant").reindex(risk_variants).reset_index()
    for col in ["improved_station_count", "worsened_station_count"]:
        if col not in risk.columns:
            risk[col] = 0
    risk[["improved_station_count", "worsened_station_count"]] = (
        risk[["improved_station_count", "worsened_station_count"]]
        .fillna(0)
        .astype(int)
    )

    x = np.arange(len(ordered_variants))
    x_risk = np.arange(len(risk_variants))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    axes[0].bar(x - width / 2, test["RMSE"].to_numpy(), width=width, label="RMSE")
    axes[0].bar(x + width / 2, test["MAE"].to_numpy(), width=width, label="MAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[v] for v in ordered_variants], rotation=20, ha="right")
    axes[0].set_ylabel("Ошибка, °C")
    axes[0].set_title("Test-ошибки по ключевым вариантам")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    axes[1].bar(x_risk - width / 2, risk["improved_station_count"].to_numpy(), width=width, label="Станций улучшено")
    axes[1].bar(x_risk + width / 2, risk["worsened_station_count"].to_numpy(), width=width, label="Станций ухудшено")
    axes[1].set_xticks(x_risk)
    axes[1].set_xticklabels([LABELS[v] for v in risk_variants], rotation=20, ha="right")
    axes[1].set_ylabel("Число станций")
    axes[1].set_title("Station-risk профиль (test)")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    out = args.output.resolve() if args.output else (run_dir / "v3_focus_compare_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
