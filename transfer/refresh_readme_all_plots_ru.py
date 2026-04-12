from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refresh_readme_plots_ru import plot_bridge_full_ru, plot_improvements_ru


MODE_RU = {
    "zero-shot": "zero-shot",
    "finetune": "дообучение",
    "scratch": "с нуля",
    "finetune+bias[station]": "дообучение + bias(station)",
    "finetune+bias[station_month]": "дообучение + bias(station+месяц)",
    "scratch+bias[station]": "с нуля + bias(station)",
    "scratch+bias[station_month]": "с нуля + bias(station+месяц)",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mode_ru(mode: str) -> str:
    return MODE_RU.get(str(mode), str(mode))


def _variant_ru(v: str) -> str:
    mapping = {
        "baseline": "Базовая линия",
        "xgb_delta_global": "XGB delta (global)",
        "xgb_delta_gated": "XGB delta + gate",
        "xgb_delta_clustered_v2": "XGB delta clustered v2",
        "xgb_delta_clustered_v2_gated": "XGB delta clustered v2 + gate",
    }
    return mapping.get(str(v), str(v))


def refresh_eda_ru() -> None:
    csv_path = Path("final_2013_2023_T_ERA5_LST_daynight.csv")
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for c in ["T", "Temperature_2m", "Dewpoint_2m", "Surface_pressure", "Total_precipitation", "Evaporation", "LST_Day", "LST_Night"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "T" not in df.columns:
        return

    # correlation matrix
    corr_cols = [c for c in ["T", "Temperature_2m", "Dewpoint_2m", "Surface_pressure", "Total_precipitation", "Evaporation", "LST_Day", "LST_Night"] if c in df.columns]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    ax.set_title("Матрица корреляций признаков")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Коэффициент корреляции")
    _save(fig, Path("eda_plots/correlation_matrix.png"))

    # temperature by year
    work = df.dropna(subset=["Date", "T"]).copy()
    if not work.empty:
        work["year"] = work["Date"].dt.year
        y = work.groupby("year")["T"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(y["year"], y["T"], marker="o", label="Средняя температура")
        ax.set_xlabel("Год")
        ax.set_ylabel("Температура, °C")
        ax.set_title("Средняя температура по годам")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, Path("eda_plots/temp_by_year.png"))

        work["month"] = work["Date"].dt.month
        m = work.groupby("month")["T"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(m["month"], m["T"], marker="o", label="Средняя температура")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Температура, °C")
        ax.set_title("Средняя температура по месяцам")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, Path("eda_plots/temp_by_month.png"))


def refresh_base_model_plots_ru() -> None:
    run_dir = Path("outputs_runs/20250916_171729_lags123_spatial")
    m_month = run_dir / "metrics_by_month_test.csv"
    m_station = run_dir / "metrics_by_station_test.csv"
    if m_month.exists():
        dfm = pd.read_csv(m_month)
        month_col = "month" if "month" in dfm.columns else ("group" if "group" in dfm.columns else None)
        if month_col is None:
            return
        dfm = dfm.sort_values(month_col)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.plot(dfm[month_col], dfm["MAE"], marker="o", label="MAE")
        if "RMSE" in dfm.columns:
            ax.plot(dfm[month_col], dfm["RMSE"], marker="o", label="RMSE")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Ошибка, °C")
        ax.set_title("Ошибка по месяцам на test")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, run_dir / "boxplot_error_by_month.png")

    if m_station.exists():
        dfs = pd.read_csv(m_station).sort_values("MAE", ascending=False).head(30)
        station_col = "station" if "station" in dfs.columns else ("group" if "group" in dfs.columns else None)
        if station_col is None:
            return
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.barh(dfs[station_col].astype(str), dfs["MAE"], label="MAE по станции")
        ax.invert_yaxis()
        ax.set_xlabel("MAE, °C")
        ax.set_ylabel("Станция")
        ax.set_title("Станции с наибольшей ошибкой на test")
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
        _save(fig, run_dir / "scatter_pred_vs_true.png")


def refresh_error_map_ru() -> None:
    run_dir = Path("outputs_runs/20250923_114911_error_map")
    src = run_dir / "station_errors_test.csv"
    if not src.exists():
        return
    dfe = pd.read_csv(src)
    station_col = "station"
    value_col = "mae"
    base_csv = Path("final_2013_2023_T_ERA5_LST_daynight.csv")
    if base_csv.exists() and {"Cod", "X_final", "Y_final"}.issubset(set(pd.read_csv(base_csv, nrows=1).columns)):
        base = pd.read_csv(base_csv, usecols=["Cod", "X_final", "Y_final"]).copy()
        base["Cod"] = base["Cod"].astype(str)
        coords = base.groupby("Cod", as_index=False)[["X_final", "Y_final"]].mean()
        merged = dfe.copy()
        merged[station_col] = merged[station_col].astype(str)
        merged = merged.merge(coords, left_on=station_col, right_on="Cod", how="left")
        if merged["X_final"].notna().sum() >= max(5, int(0.5 * len(merged))):
            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            sc = ax.scatter(
                merged["X_final"],
                merged["Y_final"],
                c=merged[value_col],
                cmap="viridis",
                s=55,
                edgecolors="black",
                linewidths=0.2,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("MAE на test, °C")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Пространственное распределение ошибки по станциям")
            ax.grid(alpha=0.2)
            _save(fig, run_dir / "map_mae_test.png")
            return

    # fallback
    top = dfe.sort_values(value_col, ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(top[station_col].astype(str), top[value_col])
    ax.invert_yaxis()
    ax.set_xlabel("MAE на test, °C")
    ax.set_ylabel("Станция")
    ax.set_title("Станции с наибольшей ошибкой")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, run_dir / "map_mae_test.png")


def refresh_ljungbox_ru() -> None:
    run_dir = Path("outputs_runs/20250923_172831_resid_acf_pacf")
    src = run_dir / "ljungbox_test.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    p_col = "p_30" if "p_30" in df.columns else ("p_14" if "p_14" in df.columns else "p_7")
    plot_df = df.sort_values(p_col, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = np.where(plot_df[p_col] < 0.05, "#b22222", "#2e8b57")
    ax.barh(plot_df["station"].astype(str), plot_df[p_col], color=colors)
    ax.axvline(0.05, color="black", linestyle="--", linewidth=1.1, label="Порог значимости 0.05")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Станция")
    ax.set_title("Ljung-Box по станциям (test)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    _save(fig, run_dir / "resid_test_winter_acf.png")


def _refresh_summary_rmse_ru(run_dir: Path) -> None:
    src = run_dir / "summary_all_cases.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if df.empty:
        return
    directions = sorted(df["direction"].dropna().astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(directions), figsize=(7.0 * len(directions), 4.4), sharey=True)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        d = df[df["direction"] == direction].copy()
        for mode, g in d.groupby("mode"):
            gg = g.sort_values("requested_target_stations")
            ax.plot(
                gg["requested_target_stations"],
                gg["RMSE"],
                marker="o",
                label=_mode_ru(str(mode)),
            )
        if direction == "east_to_west":
            dir_title = "Восток -> Запад"
        elif direction == "west_to_east":
            dir_title = "Запад -> Восток"
        else:
            dir_title = direction
        ax.set_title(dir_title)
        ax.set_xlabel("Число калибровочных станций в target")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE на test, °C")
    axes[-1].legend(frameon=False)
    _save(fig, run_dir / "summary_rmse.png")


def _refresh_rmse_by_seed_ru(run_dir: Path) -> None:
    src = run_dir / "summary_by_seed.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for mode, g in df.groupby("mode"):
        gg = g.sort_values("seed")
        ax.plot(gg["seed"], gg["RMSE"], marker="o", label=_mode_ru(str(mode)))
    ax.set_xlabel("Сид")
    ax.set_ylabel("RMSE на test, °C")
    ax.set_title("Зимний transfer: RMSE по сидам")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, run_dir / "rmse_by_seed.png")


def _refresh_loso_station_ru(run_dir: Path) -> None:
    src = run_dir / "metrics_by_station_loso.csv"
    if not src.exists():
        return
    df = pd.read_csv(src).sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(df["station"].astype(str), df["RMSE"])
    ax.set_xlabel("Станция")
    ax.set_ylabel("RMSE на test, °C")
    ax.set_title("LOSO: RMSE по удержанной станции")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, run_dir / "rmse_by_station_loso.png")


def _refresh_uncertainty_ru(run_dir: Path) -> None:
    month_csv = run_dir / "coverage_by_month_test.csv"
    pred_csv = run_dir / "predictions_test_intervals.csv"
    if month_csv.exists():
        df = pd.read_csv(month_csv)
        mcol = "group" if "group" in df.columns else "month"
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(df[mcol], df["coverage_p10_p90"], marker="o", label="Фактическое покрытие")
        ax.axhline(0.85, linestyle="--", linewidth=1.1, label="Целевое покрытие 0.85")
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Покрытие")
        ax.set_title("Покрытие интервала P10-P90 по месяцам")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, run_dir / "coverage_by_month_test.png")
    if pred_csv.exists():
        pred = pd.read_csv(pred_csv)
        if "interval_width" in pred.columns:
            widths = pd.to_numeric(pred["interval_width"], errors="coerce").dropna().to_numpy()
            if len(widths):
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                ax.hist(widths, bins=40, alpha=0.9, label="Ширина интервала")
                ax.set_xlabel("Ширина интервала (P90 - P10), °C")
                ax.set_ylabel("Частота")
                ax.set_title("Распределение ширины интервалов на test")
                ax.grid(alpha=0.25)
                ax.legend()
                _save(fig, run_dir / "interval_width_hist_test.png")


def _refresh_winter_hybrid_ru(run_dir: Path) -> None:
    src = run_dir / "mae_by_month_comparison.csv"
    if not src.exists():
        return
    df = pd.read_csv(src).sort_values("month")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(df["month"], df["MAE_baseline"], marker="o", label="Базовая full-модель")
    ax.plot(df["month"], df["MAE_hybrid"], marker="o", label="Гибрид full + winter")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Месяц")
    ax.set_ylabel("MAE на test, °C")
    ax.set_title("Помесячный MAE: baseline vs winter-hybrid")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, run_dir / "mae_by_month_comparison.png")


def _refresh_winter_weight_scan_ru(run_dir: Path) -> None:
    src = run_dir / "summary_scan.csv"
    if not src.exists():
        return
    df = pd.read_csv(src).sort_values("factor")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(df["factor"], df["RMSE_full"], marker="o", label="RMSE (весь test)")
    if "RMSE_winter" in df.columns:
        ax.plot(df["factor"], df["RMSE_winter"], marker="o", label="RMSE (зима)")
    ax.set_xlabel("Коэффициент веса зимы")
    ax.set_ylabel("RMSE, °C")
    ax.set_title("Скан зимнего веса")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, run_dir / "rmse_scan.png")


def _refresh_w2e_weight_summary_ru(run_dir: Path) -> None:
    src = run_dir / "best_by_weight_budget.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for budget, g in df.groupby("target_station_count"):
        gg = g.sort_values("winter_weight_factor")
        ax.plot(
            gg["winter_weight_factor"],
            gg["RMSE"],
            marker="o",
            label=f"target stations = {int(budget)}",
        )
    ax.set_xlabel("Коэффициент зимнего веса")
    ax.set_ylabel("Лучший RMSE, °C")
    ax.set_title("Лучший RMSE по бюджету станций и зимнему весу")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, run_dir / "rmse_best_vs_winter_weight.png")


def refresh_section_8_ru() -> None:
    bridge_run = Path("outputs_runs/20260411_195225_rp5_hydromet_bridge_full_selected125")
    if bridge_run.exists():
        plot_bridge_full_ru(bridge_run)

    for run in [
        Path("outputs_runs/20260411_201020_rp5_hydromet_bridge_improvements_selected125"),
        Path("outputs_runs/20260411_214201_rp5_hydromet_bridge_improvements_selected125"),
        Path("outputs_runs/20260412_conformal_grid/g4_m10"),
        Path("outputs_runs/20260412_130500_bridge_expanded_min10_v2"),
    ]:
        if run.exists():
            plot_improvements_ru(run)

    # stability charts for runs used in README
    for run in [
        Path("outputs_runs/20260412_123500_rp5_hydromet_bridge_improvements_clustered"),
    ]:
        yearly_csv = run / "metrics_by_test_year.csv"
        risk_csv = run / "station_risk_summary_test.csv"
        if not (yearly_csv.exists() and risk_csv.exists()):
            continue
        yearly = pd.read_csv(yearly_csv)
        risk = pd.read_csv(risk_csv)

        variants = ["baseline", "xgb_delta_global", "xgb_delta_gated", "xgb_delta_clustered_v2", "xgb_delta_clustered_v2_gated"]
        y = yearly[yearly["variant"].isin(variants)].copy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for v in variants:
            g = y[y["variant"] == v].sort_values("year")
            if g.empty:
                continue
            label = _variant_ru(v)
            axes[0].plot(g["year"], g["RMSE"], marker="o", label=label)
            axes[1].plot(g["year"], g["MAE"], marker="o", label=label)
        axes[0].set_xlabel("Год")
        axes[0].set_ylabel("RMSE, °C")
        axes[0].set_title("Устойчивость по годам: RMSE")
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Год")
        axes[1].set_ylabel("MAE, °C")
        axes[1].set_title("Устойчивость по годам: MAE")
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        _save(fig, run / "stability_by_year_rmse_mae.png")

        r = risk[risk["variant"].isin(["xgb_delta_global", "xgb_delta_gated", "xgb_delta_clustered_v2", "xgb_delta_clustered_v2_gated"])].copy()
        if not r.empty:
            r = r.sort_values("worsened_station_count")
            fig, ax = plt.subplots(figsize=(9, 4.8))
            x = np.arange(len(r))
            ax.bar(x - 0.2, r["worsened_station_count"], width=0.4, label="Станции с ухудшением")
            ax.bar(x + 0.2, r["improved_station_count"], width=0.4, label="Станции с улучшением")
            ax.set_xticks(x)
            ax.set_xticklabels([_variant_ru(v) for v in r["variant"]], rotation=20, ha="right")
            ax.set_xlabel("Вариант модели")
            ax.set_ylabel("Число станций")
            ax.set_title("Риск по станциям на тесте")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            _save(fig, run / "station_risk_improved_vs_worsened.png")


def main() -> None:
    refresh_eda_ru()
    refresh_base_model_plots_ru()
    refresh_error_map_ru()
    refresh_ljungbox_ru()

    for rd in [
        Path("outputs_runs/20260327_171818_spatial_transfer_preflight"),
        Path("outputs_runs/20260407_164430_spatial_transfer_preflight_serious_fix"),
        Path("outputs_runs/20260407_185400_spatial_transfer_w2e_bias"),
    ]:
        _refresh_summary_rmse_ru(rd)

    for rd in [
        Path("outputs_runs/20260407_161100_volgograd_winter_multiseed_full5"),
        Path("outputs_runs/20260407_164430_volgograd_winter_multiseed_x10"),
    ]:
        _refresh_rmse_by_seed_ru(rd)

    _refresh_loso_station_ru(Path("outputs_runs/20260407_160800_saratov_loso_full14"))

    for rd in [
        Path("outputs_runs/20260407_161900_saratov_uncertainty_full_calibrated"),
        Path("outputs_runs/20260407_164430_saratov_uncertainty_cov85_strict"),
        Path("outputs_runs/20260407_184700_saratov_uncertainty_cov85_conformal_holdout"),
    ]:
        _refresh_uncertainty_ru(rd)

    _refresh_winter_hybrid_ru(Path("outputs_runs/20260407_163025_saratov_winter_hybrid"))
    _refresh_winter_weight_scan_ru(Path("outputs_runs/20260407_163846_saratov_winter_weight_scan"))
    _refresh_w2e_weight_summary_ru(Path("outputs_runs/20260407_194700_w2e_stationmonth_weight_scan_summary"))

    refresh_section_8_ru()
    print("Refreshed README plots with Russian labels where source CSV artifacts are available.")


if __name__ == "__main__":
    main()
