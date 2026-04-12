from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "baseline": "Базовая линия RP5",
    "xgb_delta_global": "XGB delta (global)",
    "xgb_delta_gated": "XGB delta + gate",
    "xgb_delta_gated_adaptive": "XGB delta + адаптивный gate",
    "xgb_delta_gated_adaptive_safeguard": "XGB delta + адаптивный gate + safeguard",
    "xgb_delta_clustered_v2_gated": "Кластерный v2 + gate",
    "xgb_delta_clustered_v3_gated": "Кластерный v3 + gate",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Построение русских stability-графиков для bridge run.")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    yearly_csv = run_dir / "metrics_by_test_year.csv"
    risk_csv = run_dir / "station_risk_summary_test.csv"
    if not yearly_csv.exists() or not risk_csv.exists():
        raise RuntimeError("Ожидаются файлы metrics_by_test_year.csv и station_risk_summary_test.csv")

    yearly = pd.read_csv(yearly_csv)
    risk = pd.read_csv(risk_csv)

    variants = [
        "baseline",
        "xgb_delta_global",
        "xgb_delta_gated",
        "xgb_delta_gated_adaptive",
        "xgb_delta_gated_adaptive_safeguard",
        "xgb_delta_clustered_v2_gated",
        "xgb_delta_clustered_v3_gated",
    ]
    y = yearly[yearly["variant"].isin(variants)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for v in variants:
        g = y[y["variant"] == v].sort_values("year")
        if g.empty:
            continue
        ax.plot(g["year"], g["RMSE"], marker="o", label=LABELS.get(v, v))
    ax.set_xlabel("Год")
    ax.set_ylabel("RMSE, °C")
    ax.set_title("Устойчивость по годам: RMSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    for v in variants:
        g = y[y["variant"] == v].sort_values("year")
        if g.empty:
            continue
        ax.plot(g["year"], g["MAE"], marker="o", label=LABELS.get(v, v))
    ax.set_xlabel("Год")
    ax.set_ylabel("MAE, °C")
    ax.set_title("Устойчивость по годам: MAE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / "stability_by_year_rmse_mae.png", dpi=140)
    plt.close(fig)

    r = risk[risk["variant"].isin(variants[1:])].copy()
    if not r.empty:
        r = r.sort_values("worsened_station_count")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(r))
        ax.bar(x - 0.2, r["worsened_station_count"], width=0.4, label="Станции с ухудшением")
        ax.bar(x + 0.2, r["improved_station_count"], width=0.4, label="Станции с улучшением")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(str(v), str(v)) for v in r["variant"]], rotation=20, ha="right")
        ax.set_xlabel("Вариант модели")
        ax.set_ylabel("Число станций")
        ax.set_title("Риск по станциям на тесте")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "station_risk_improved_vs_worsened.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
