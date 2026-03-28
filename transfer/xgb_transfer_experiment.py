from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from pipeline_common import (
    TARGET_COLUMN,
    build_feature_frame,
    choose_validation_year,
    evaluate_model,
    filter_winter,
    limit_to_station_subset,
    load_dataset,
    resolve_feature_list,
    save_json,
    save_run_bundle,
    split_by_year,
    train_xgb,
    tune_xgb,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Перенос XGBoost-модели между регионами")
    parser.add_argument("--source-csv", required=True, help="Исходный регион, например Саратов")
    parser.add_argument("--target-csv", required=True, help="Целевой регион, например Волгоград")
    parser.add_argument("--modes", nargs="+", default=["zero-shot", "finetune", "scratch"], choices=["zero-shot", "finetune", "scratch"])
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--test-end-year", type=int, default=2023)
    parser.add_argument("--target-max-stations", type=int, default=None, help="Ограничение числа станций для проверки режима малой сети")
    parser.add_argument("--zero-inflated-precip", action="store_true")
    parser.add_argument("--winter-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=4000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser


def prepare_dataset(
    csv_path: str,
    *,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    zero_inflated_precip: bool,
    include_station_mean: bool,
    winter_only: bool,
    max_stations: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, str, dict[str, pd.DataFrame], list[str]]:
    df, meta = load_dataset(csv_path)
    train_mask, test_mask = split_by_year(
        df.assign(year=pd.to_datetime(df[meta.date_col]).dt.year),
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    )
    df = build_feature_frame(
        df,
        meta,
        train_mask=train_mask,
        zero_inflated_precip=zero_inflated_precip,
        include_station_mean=include_station_mean,
    )
    if winter_only:
        df = filter_winter(df)
    if max_stations is not None:
        df = limit_to_station_subset(df, meta.station_col, max_stations=max_stations, seed=seed)

    train_mask, test_mask = split_by_year(
        df,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    )
    train = df.loc[train_mask].dropna(subset=[TARGET_COLUMN]).copy()
    test = df.loc[test_mask].dropna(subset=[TARGET_COLUMN]).copy()
    if train.empty or test.empty:
        raise RuntimeError(f"После split выборки пусты для {csv_path}")

    val_year = choose_validation_year(train)
    inner_train = train[train["year"] < val_year].copy()
    inner_val = train[train["year"] == val_year].copy()
    if inner_train.empty or inner_val.empty:
        raise RuntimeError(f"Не удалось отделить inner_train/inner_val для {csv_path}")

    features = resolve_feature_list(df, include_station_mean=include_station_mean)
    splits = {
        "train": train,
        "test": test,
        "inner_train": inner_train,
        "inner_val": inner_val,
        "full": df.dropna(subset=[TARGET_COLUMN]).copy(),
    }
    return df, meta.station_col, splits, features


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.output_dir or f"outputs_runs/{ts}_transfer_region"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(
        f"[transfer] start outdir={outdir} modes={args.modes} target_max_stations={args.target_max_stations} device={args.device} n_trials={args.n_trials}",
        flush=True,
    )

    source_df, _, source_splits, source_features = prepare_dataset(
        args.source_csv,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        zero_inflated_precip=args.zero_inflated_precip,
        include_station_mean=False,
        winter_only=args.winter_only,
        seed=args.seed,
    )
    target_df, target_station_col, target_splits_nom, target_features_nom = prepare_dataset(
        args.target_csv,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        zero_inflated_precip=args.zero_inflated_precip,
        include_station_mean=False,
        winter_only=args.winter_only,
        max_stations=args.target_max_stations,
        seed=args.seed,
    )
    _, _, target_splits_mean, target_features_mean = prepare_dataset(
        args.target_csv,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        zero_inflated_precip=args.zero_inflated_precip,
        include_station_mean=True,
        winter_only=args.winter_only,
        max_stations=args.target_max_stations,
        seed=args.seed,
    )

    common_nominal_features = sorted(set(source_features).intersection(target_features_nom))
    if not common_nominal_features:
        raise RuntimeError("Нет общих признаков между source и target для zero-shot/finetune")
    print(
        f"[transfer] prepared source_rows={len(source_df)} target_rows={len(target_df)} nominal_features={len(common_nominal_features)} target_features_mean={len(target_features_mean)}",
        flush=True,
    )

    source_params = tune_xgb(
        source_splits["inner_train"],
        source_splits["inner_val"],
        common_nominal_features,
        device=args.device,
        n_trials=args.n_trials,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        progress_label="source_tune",
    )
    source_model = train_xgb(
        source_splits["train"],
        source_splits["inner_val"],
        common_nominal_features,
        source_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=100,
        progress_label="source_train",
    )

    summary_rows: list[dict[str, float | int | str]] = []

    if "zero-shot" in args.modes:
        print("[transfer] mode=zero-shot start", flush=True)
        zero_preds_test, zero_metrics_test = evaluate_model(source_model, target_splits_nom["test"], common_nominal_features)
        zero_preds_full, zero_metrics_full = evaluate_model(source_model, target_splits_nom["full"], common_nominal_features)
        mode_outdir = Path(outdir) / "zero_shot"
        save_run_bundle(
            mode_outdir,
            metrics={"test": zero_metrics_test, "full": zero_metrics_full},
            features=common_nominal_features,
            params=source_params,
            predictions={
                "test": (target_splits_nom["test"], zero_preds_test),
                "full": (target_splits_nom["full"], zero_preds_full),
            },
            station_col=target_station_col,
        )
        summary_rows.append({"mode": "zero-shot", **zero_metrics_test})
        print(f"[transfer] mode=zero-shot done test_metrics={zero_metrics_test}", flush=True)

    if "scratch" in args.modes:
        print("[transfer] mode=scratch start", flush=True)
        scratch_params = tune_xgb(
            target_splits_mean["inner_train"],
            target_splits_mean["inner_val"],
            target_features_mean,
            device=args.device,
            n_trials=args.n_trials,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed,
            progress_label="scratch_tune",
        )
        scratch_model = train_xgb(
            target_splits_mean["train"],
            target_splits_mean["inner_val"],
            target_features_mean,
            scratch_params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=100,
            progress_label="scratch_train",
        )
        scratch_preds_train, scratch_metrics_train = evaluate_model(scratch_model, target_splits_mean["train"], target_features_mean)
        scratch_preds_test, scratch_metrics_test = evaluate_model(scratch_model, target_splits_mean["test"], target_features_mean)
        scratch_preds_full, scratch_metrics_full = evaluate_model(scratch_model, target_splits_mean["full"], target_features_mean)
        mode_outdir = Path(outdir) / "scratch"
        save_run_bundle(
            mode_outdir,
            metrics={"train": scratch_metrics_train, "test": scratch_metrics_test, "full": scratch_metrics_full},
            features=target_features_mean,
            params=scratch_params,
            predictions={
                "train": (target_splits_mean["train"], scratch_preds_train),
                "test": (target_splits_mean["test"], scratch_preds_test),
                "full": (target_splits_mean["full"], scratch_preds_full),
            },
            model=scratch_model,
            station_col=target_station_col,
        )
        summary_rows.append({"mode": "scratch", **scratch_metrics_test})
        print(f"[transfer] mode=scratch done test_metrics={scratch_metrics_test}", flush=True)

    if "finetune" in args.modes:
        print("[transfer] mode=finetune start", flush=True)
        finetune_params = dict(source_params)
        finetune_params["learning_rate"] = min(float(source_params["learning_rate"]), 0.03)
        finetune_model = train_xgb(
            target_splits_nom["train"],
            target_splits_nom["inner_val"],
            common_nominal_features,
            finetune_params,
            num_boost_round=max(800, args.num_boost_round // 2),
            early_stopping_rounds=max(50, args.early_stopping_rounds // 2),
            base_model=source_model,
            verbose_eval=50,
            progress_label="finetune_train",
        )
        finetune_preds_test, finetune_metrics_test = evaluate_model(finetune_model, target_splits_nom["test"], common_nominal_features)
        finetune_preds_full, finetune_metrics_full = evaluate_model(finetune_model, target_splits_nom["full"], common_nominal_features)
        mode_outdir = Path(outdir) / "finetune"
        save_run_bundle(
            mode_outdir,
            metrics={"test": finetune_metrics_test, "full": finetune_metrics_full},
            features=common_nominal_features,
            params=finetune_params,
            predictions={
                "test": (target_splits_nom["test"], finetune_preds_test),
                "full": (target_splits_nom["full"], finetune_preds_full),
            },
            model=finetune_model,
            station_col=target_station_col,
        )
        summary_rows.append({"mode": "finetune", **finetune_metrics_test})
        print(f"[transfer] mode=finetune done test_metrics={finetune_metrics_test}", flush=True)

    pd.DataFrame(summary_rows).to_csv(Path(outdir) / "summary_metrics.csv", index=False)
    save_json(
        Path(outdir) / "run_config.json",
        {
            "source_csv": args.source_csv,
            "target_csv": args.target_csv,
            "modes": args.modes,
            "train_years": [args.train_start_year, args.train_end_year],
            "test_years": [args.test_start_year, args.test_end_year],
            "target_max_stations": args.target_max_stations,
            "zero_inflated_precip": args.zero_inflated_precip,
            "winter_only": args.winter_only,
            "device": args.device,
            "n_trials": args.n_trials,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
            "source_rows": int(len(source_df)),
            "target_rows": int(len(target_df)),
        },
    )
    print(f"[transfer] summary_rows={summary_rows}", flush=True)
    print(f"Saved transfer run: {outdir}")


if __name__ == "__main__":
    main()
