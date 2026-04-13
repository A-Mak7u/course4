from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Собрать одну таблицу сравнения вариантов с эталоном xgb_delta_gated."
    )
    p.add_argument("--control-run-dir", required=True, help="Папка контрольного run с variant_metrics.csv")
    p.add_argument("--improved-run-dir", required=True, help="Папка улучшенного run с variant_metrics.csv")
    p.add_argument("--etalon-variant", default="xgb_delta_gated")
    p.add_argument(
        "--variants",
        default="baseline,xgb_delta_gated,xgb_delta_gated_adaptive,xgb_delta_gated_adaptive_safeguard,xgb_delta_selector_station_month",
        help="Список вариантов через запятую, которые оставить в таблице.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Куда сохранить таблицу. По умолчанию: <improved-run-dir>/compare_vs_etalon_table.csv",
    )
    return p.parse_args()


def load_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "variant_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise RuntimeError(f"В {path} нет столбца split")
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise RuntimeError(f"В {path} нет test-метрик")
    return test


def main() -> None:
    args = parse_args()
    control_run = Path(args.control_run_dir)
    improved_run = Path(args.improved_run_dir)
    selected = [v.strip() for v in str(args.variants).split(",") if v.strip()]

    control_test = load_metrics(control_run)
    improved_test = load_metrics(improved_run)

    etalon = control_test[control_test["variant"] == args.etalon_variant]
    if etalon.empty:
        raise RuntimeError(f"В control run не найден эталонный вариант: {args.etalon_variant}")
    etalon_row = etalon.iloc[0]

    baseline_ctrl = control_test[control_test["variant"] == "baseline"].iloc[0]
    baseline_imp = improved_test[improved_test["variant"] == "baseline"].iloc[0]

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "run_tag": "control_etalon",
            "variant": str(args.etalon_variant),
            "RMSE": float(etalon_row["RMSE"]),
            "MAE": float(etalon_row["MAE"]),
            "R2": float(etalon_row["R2"]),
            "RMSE_gain_vs_baseline_run": float(baseline_ctrl["RMSE"] - etalon_row["RMSE"]),
            "MAE_gain_vs_baseline_run": float(baseline_ctrl["MAE"] - etalon_row["MAE"]),
            "RMSE_gain_vs_control_etalon": 0.0,
            "MAE_gain_vs_control_etalon": 0.0,
        }
    )

    for _, r in improved_test.iterrows():
        variant = str(r["variant"])
        if selected and variant not in selected:
            continue
        rows.append(
            {
                "run_tag": "improved",
                "variant": variant,
                "RMSE": float(r["RMSE"]),
                "MAE": float(r["MAE"]),
                "R2": float(r["R2"]),
                "RMSE_gain_vs_baseline_run": float(baseline_imp["RMSE"] - r["RMSE"]),
                "MAE_gain_vs_baseline_run": float(baseline_imp["MAE"] - r["MAE"]),
                "RMSE_gain_vs_control_etalon": float(etalon_row["RMSE"] - r["RMSE"]),
                "MAE_gain_vs_control_etalon": float(etalon_row["MAE"] - r["MAE"]),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["run_tag", "RMSE"], ascending=[True, False]).reset_index(drop=True)

    out_path = Path(args.output_csv) if args.output_csv else (improved_run / "compare_vs_etalon_table.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
