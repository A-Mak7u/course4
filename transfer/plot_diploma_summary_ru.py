from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построение сводных графиков для диплома (bridge + transfer)."
    )
    parser.add_argument(
        "--control-expanded-csv",
        default="reports/bridge/control_vs_expanded_post_anti_leak_v2.csv",
    )
    parser.add_argument(
        "--adaptive-vs-etalon-csv",
        default="reports/bridge/adaptive_safeguard_vs_etalon_post_anti_leak.csv",
    )
    parser.add_argument(
        "--transfer-summary-csv",
        default="outputs_runs/20260412_141500_transfer_volgograd_v2/transfer_modes_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
    )
    return parser.parse_args()


def _load_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {name}: {path}")
    return pd.read_csv(path)


def plot_bridge_key_metrics(df: pd.DataFrame, outdir: Path) -> None:
    key_variants = ["baseline", "xgb_delta_gated", "xgb_delta_selector_station"]
    key = df[df["variant"].isin(key_variants)].copy()
    if key.empty:
        return

    dataset_label = {
        "control_selected125_2013_2023": "Контроль (125 станций)",
        "expanded_min10_2013_2023": "Расширенный (132 станции)",
    }
    variant_label = {
        "baseline": "Базовая линия (RP5)",
        "xgb_delta_gated": "XGB delta + gate",
        "xgb_delta_selector_station": "XGB delta meta-selector",
    }
    key["dataset_ru"] = key["dataset"].map(dataset_label).fillna(key["dataset"])
    key["variant_ru"] = key["variant"].map(variant_label).fillna(key["variant"])

    datasets = ["Контроль (125 станций)", "Расширенный (132 станции)"]
    variants = [variant_label[v] for v in key_variants]
    x = np.arange(len(datasets))
    w = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for i, vv in enumerate(variants):
        sub = key[key["variant_ru"] == vv].set_index("dataset_ru").reindex(datasets)
        axes[0].bar(x + (i - 1) * w, sub["RMSE"].values, width=w, label=vv)
        axes[1].bar(x + (i - 1) * w, sub["MAE"].values, width=w, label=vv)

    for ax, metric in zip(axes, ["RMSE, °C", "MAE, °C"]):
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=0)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_title("Сравнение RMSE по ключевым вариантам")
    axes[1].set_title("Сравнение MAE по ключевым вариантам")
    fig.tight_layout()
    fig.savefig(outdir / "diploma_bridge_key_metrics_selected_vs_expanded.png", dpi=160)
    plt.close(fig)


def plot_bridge_risk_profile(df: pd.DataFrame, outdir: Path) -> None:
    want = [
        "xgb_delta_gated",
        "xgb_delta_gated_adaptive_safeguard",
        "xgb_delta_selector_station",
    ]
    sub = df[(df["dataset"] == "expanded_min10_2013_2023") & (df["variant"].isin(want))].copy()
    if sub.empty:
        return

    ru = {
        "xgb_delta_gated": "XGB delta + gate",
        "xgb_delta_gated_adaptive_safeguard": "XGB delta + adaptive+safeguard",
        "xgb_delta_selector_station": "XGB delta meta-selector",
    }
    sub["variant_ru"] = sub["variant"].map(ru).fillna(sub["variant"])
    sub = sub.sort_values("variant_ru")

    x = np.arange(len(sub))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(x - w / 2, sub["improved_station_count"], width=w, label="Улучшенные станции")
    ax.bar(x + w / 2, sub["worsened_station_count"], width=w, label="Ухудшенные станции")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["variant_ru"], rotation=0)
    ax.set_ylabel("Количество станций")
    ax.set_title("Профиль station-risk на expanded-наборе")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "diploma_bridge_station_risk_expanded.png", dpi=160)
    plt.close(fig)


def plot_transfer_modes(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty or "mode" not in df.columns:
        return

    mode_order = ["zero-shot", "finetune", "scratch"]
    ru = {
        "zero-shot": "Zero-shot",
        "finetune": "Finetune",
        "scratch": "Scratch",
    }
    data = df.copy()
    data["mode"] = data["mode"].astype(str)
    data = data[data["mode"].isin(mode_order)].set_index("mode").reindex(mode_order).reset_index()
    data["mode_ru"] = data["mode"].map(ru).fillna(data["mode"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].bar(data["mode_ru"], data["RMSE"], color=["#d9534f", "#f0ad4e", "#5cb85c"])
    axes[0].set_ylabel("RMSE, °C")
    axes[0].set_title("Волгоград: RMSE по режимам переноса")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(data["mode_ru"], data["MAE"], color=["#d9534f", "#f0ad4e", "#5cb85c"])
    axes[1].set_ylabel("MAE, °C")
    axes[1].set_title("Волгоград: MAE по режимам переноса")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "diploma_transfer_modes_volgograd.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    control_expanded = _load_required(Path(args.control_expanded_csv), "control_vs_expanded")
    _ = _load_required(Path(args.adaptive_vs_etalon_csv), "adaptive_vs_etalon")
    transfer = _load_required(Path(args.transfer_summary_csv), "transfer_summary")

    plot_bridge_key_metrics(control_expanded, outdir)
    plot_bridge_risk_profile(control_expanded, outdir)
    plot_transfer_modes(transfer, outdir)

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
