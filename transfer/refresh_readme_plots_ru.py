from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS_RU = {
    "global_quantile": "Глобальный квантиль",
    "monthly_conformal": "Помесячный conformal",
    "conditional_station_group_month": "Условный conformal (группа станции + месяц)",
}

VARIANT_LABELS_RU = {
    "baseline": "Базовая линия (RP5)",
    "ridge_global": "Ridge (global)",
    "ridge_gated": "Ridge + gate",
    "ridge_gated_station_month": "Ridge + gate(station+month)",
    "ridge_soft_station": "Ridge soft(station)",
    "ridge_soft_station_month": "Ridge soft(station+month)",
    "ridge_seasonal": "Ridge seasonal",
    "ridge_downweight": "Ridge downweight",
    "xgb_global": "XGBoost (global)",
    "xgb_gated": "XGBoost + gate",
    "xgb_gated_station_month": "XGBoost + gate(station+month)",
    "xgb_soft_station": "XGBoost soft(station)",
    "xgb_soft_station_month": "XGBoost soft(station+month)",
    "xgb_delta_global": "XGBoost delta (global)",
    "xgb_delta_gated": "XGBoost delta + gate",
    "xgb_delta_gated_station_month": "XGBoost delta + gate(station+month)",
    "xgb_delta_clustered": "XGBoost delta clustered",
    "xgb_delta_clustered_gated": "XGBoost delta clustered + gate",
    "xgb_delta_clustered_v2": "XGBoost delta clustered v2",
    "xgb_delta_clustered_v2_gated": "XGBoost delta clustered v2 + gate",
}


def _variant_ru(name: str) -> str:
    return VARIANT_LABELS_RU.get(str(name), str(name))


def _save(fig: plt.Figure, path: Path, *, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if close:
        plt.close(fig)


def plot_bridge_full_ru(run_dir: Path) -> None:
    pred_csv = run_dir / "bridge_predictions.csv"
    monthly_csv = run_dir / "metrics_by_month.csv"
    station_csv = run_dir / "metrics_by_station.csv"
    if not pred_csv.exists():
        return

    df = pd.read_csv(pred_csv)
    monthly_df = pd.read_csv(monthly_csv) if monthly_csv.exists() else pd.DataFrame()
    station_df = pd.read_csv(station_csv) if station_csv.exists() else pd.DataFrame()

    if {"T_rp5", "T_hydromet"}.issubset(df.columns):
        rp5_vals = pd.to_numeric(df["T_rp5"], errors="coerce").to_numpy()
        true_vals = pd.to_numeric(df["T_hydromet"], errors="coerce").to_numpy()
        mask = np.isfinite(rp5_vals) & np.isfinite(true_vals)
        rp5_vals = rp5_vals[mask]
        true_vals = true_vals[mask]
        if len(rp5_vals):
            lo = float(min(rp5_vals.min(), true_vals.min()))
            hi = float(max(rp5_vals.max(), true_vals.max()))
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(rp5_vals, true_vals, s=8, alpha=0.25, edgecolors="none", label="Наблюдения")
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Идеальное совпадение")
            ax.set_xlabel("Температура RP5, °C")
            ax.set_ylabel("Температура Росгидромета, °C")
            ax.set_title("RP5 и Росгидромет: сравнение по точкам")
            ax.grid(alpha=0.2)
            ax.legend()
            _save(fig, run_dir / "rp5_hydromet_scatter_xy.png", close=False)
            _save(fig, run_dir / "scatter_rp5_vs_hydromet.png")

            delta = rp5_vals - true_vals
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(delta, bins=60, alpha=0.9, label="Распределение разности")
            ax.axvline(0.0, linestyle="--", linewidth=1.2, label="Нулевая разность")
            ax.set_xlabel("Разность температур (RP5 - Росгидромет), °C")
            ax.set_ylabel("Частота")
            ax.set_title("Распределение разности температур")
            ax.grid(alpha=0.2)
            ax.legend()
            _save(fig, run_dir / "delta_hist.png")

    if not monthly_df.empty and {"month", "baseline_mae", "bridge_mae"}.issubset(monthly_df.columns):
        m = monthly_df.copy().sort_values("month")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(m["month"], m["baseline_mae"], marker="o", label="Базовая линия (RP5)")
        ax.plot(m["month"], m["bridge_mae"], marker="o", label="Калибровочный мост")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Месяц")
        ax.set_ylabel("MAE, °C")
        ax.set_title("MAE по месяцам")
        ax.grid(alpha=0.2)
        ax.legend()
        _save(fig, run_dir / "delta_mae_by_month.png")

        m["mae_gain"] = m["baseline_mae"] - m["bridge_mae"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#2e8b57" if v >= 0 else "#b22222" for v in m["mae_gain"]]
        ax.bar(m["month"], m["mae_gain"], color=colors)
        ax.axhline(0.0, linestyle="--", linewidth=1.1, color="black")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Прирост MAE (база - мост), °C")
        ax.set_title("Прирост качества по месяцам")
        ax.grid(axis="y", alpha=0.2)
        _save(fig, run_dir / "delta_mae_gain_by_month.png")

    if not station_df.empty and {"station", "baseline_mae", "bridge_mae"}.issubset(station_df.columns):
        s = station_df.copy()
        s["mae_gain"] = s["baseline_mae"] - s["bridge_mae"]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(s["mae_gain"], bins=40, alpha=0.9, label="Станции")
        ax.axvline(0.0, linestyle="--", linewidth=1.1, color="black", label="Нулевой прирост")
        ax.set_xlabel("Прирост MAE по станции (база - мост), °C")
        ax.set_ylabel("Частота")
        ax.set_title("Распределение station-wise прироста")
        ax.grid(alpha=0.2)
        ax.legend()
        _save(fig, run_dir / "station_mae_gain_hist.png")

        tail = s.sort_values("mae_gain").head(10)
        head = s.sort_values("mae_gain").tail(10)
        mix = pd.concat([tail, head], axis=0).copy()
        mix = mix.sort_values("mae_gain")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#b22222" if v < 0 else "#2e8b57" for v in mix["mae_gain"]]
        y = np.arange(len(mix))
        ax.barh(y, mix["mae_gain"], color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(mix["station"].astype(str))
        ax.axvline(0.0, linestyle="--", linewidth=1.1, color="black")
        ax.set_xlabel("Прирост MAE (база - мост), °C")
        ax.set_ylabel("Станция")
        ax.set_title("Станции с наименьшим и наибольшим приростом")
        ax.grid(axis="x", alpha=0.2)
        _save(fig, run_dir / "station_mae_top20_tail.png")


def plot_improvements_ru(run_dir: Path) -> None:
    metrics_csv = run_dir / "variant_metrics.csv"
    if metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)
        test = metrics[metrics["split"] == "test"].copy()
        if not test.empty:
            order = test.sort_values("RMSE")["variant"].tolist()

            fig, ax = plt.subplots(figsize=(10, 5))
            vals = test.set_index("variant").loc[order, "RMSE"]
            ax.bar(np.arange(len(order)), vals.values)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([_variant_ru(v) for v in order], rotation=30, ha="right")
            ax.set_xlabel("Вариант модели")
            ax.set_ylabel("RMSE, °C")
            ax.set_title("Сравнение вариантов на test (2022-2023): RMSE")
            ax.grid(axis="y", alpha=0.2)
            _save(fig, run_dir / "variant_rmse_test.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            vals = test.set_index("variant").loc[order, "MAE"]
            ax.bar(np.arange(len(order)), vals.values)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([_variant_ru(v) for v in order], rotation=30, ha="right")
            ax.set_xlabel("Вариант модели")
            ax.set_ylabel("MAE, °C")
            ax.set_title("Сравнение вариантов на test (2022-2023): MAE")
            ax.grid(axis="y", alpha=0.2)
            _save(fig, run_dir / "variant_mae_test.png")

    intervals_csv = run_dir / "intervals_summary.csv"
    if intervals_csv.exists():
        interval_global = pd.read_csv(intervals_csv)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, g in interval_global.groupby("method"):
            gg = g.sort_values("target_coverage")
            label = METHOD_LABELS_RU.get(str(method), str(method))
            ax.plot(gg["target_coverage"], gg["achieved_coverage"], marker="o", label=label)
        ax.plot([0.75, 0.95], [0.75, 0.95], linestyle="--", linewidth=1.1, label="Идеальное совпадение")
        ax.set_xlabel("Целевое покрытие")
        ax.set_ylabel("Фактическое покрытие")
        ax.set_title("Калибровка интервалов неопределённости")
        ax.grid(alpha=0.2)
        ax.legend()
        _save(fig, run_dir / "intervals_target_vs_achieved.png")

    monthly_csv = run_dir / "intervals_by_month.csv"
    if monthly_csv.exists():
        monthly = pd.read_csv(monthly_csv)
        focus_cov = 0.85
        g = monthly[monthly["target_coverage"] == focus_cov].sort_values("month")
        if not g.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(g["month"], g["coverage"], marker="o", label="Покрытие по месяцам")
            ax.axhline(focus_cov, linestyle="--", linewidth=1.1, label="Целевое покрытие 0.85")
            ax.set_xticks(range(1, 13))
            ax.set_xlabel("Месяц")
            ax.set_ylabel("Покрытие")
            ax.set_title("Покрытие monthly conformal по месяцам")
            ax.grid(alpha=0.2)
            ax.legend()
            _save(fig, run_dir / "intervals_monthly_coverage_085.png")


def main() -> None:
    bridge_run = Path("outputs_runs/20260411_195225_rp5_hydromet_bridge_full_selected125")
    improvements_runs = [
        Path("outputs_runs/20260411_201020_rp5_hydromet_bridge_improvements_selected125"),
        Path("outputs_runs/20260411_214201_rp5_hydromet_bridge_improvements_selected125"),
        Path("outputs_runs/20260412_conformal_grid/g4_m10"),
    ]

    if bridge_run.exists():
        plot_bridge_full_ru(bridge_run)
        print(f"refreshed bridge plots: {bridge_run}")

    for run in improvements_runs:
        if run.exists():
            plot_improvements_ru(run)
            print(f"refreshed improvements plots: {run}")


if __name__ == "__main__":
    main()
