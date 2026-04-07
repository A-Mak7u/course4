from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Мульти-сидовый запуск winter-only transfer (Saratov -> target region)"
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--modes", nargs="+", default=["zero-shot", "finetune", "scratch"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--output-dir", default=None)
    return parser


def run_one_seed(args: argparse.Namespace, seed: int, out_root: Path) -> Path:
    seed_outdir = out_root / f"seed_{seed}"
    seed_outdir.mkdir(parents=True, exist_ok=True)
    log_path = seed_outdir / "run.log"

    cmd = [
        sys.executable,
        "transfer/xgb_transfer_experiment.py",
        "--source-csv",
        args.source_csv,
        "--target-csv",
        args.target_csv,
        "--output-dir",
        str(seed_outdir),
        "--device",
        args.device,
        "--n-trials",
        str(args.n_trials),
        "--num-boost-round",
        str(args.num_boost_round),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
        "--seed",
        str(seed),
        "--winter-only",
        "--modes",
    ] + list(args.modes)

    print(f"[winter-multiseed] seed={seed} start outdir={seed_outdir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"seed={seed} failed, see {log_path}")
    print(f"[winter-multiseed] seed={seed} done", flush=True)
    return seed_outdir


def plot_rmse_by_seed(summary: pd.DataFrame, output_png: Path) -> None:
    if summary.empty:
        return
    plt.figure(figsize=(8, 4.5))
    for mode, group in summary.groupby("mode"):
        group = group.sort_values("seed")
        plt.plot(group["seed"], group["RMSE"], marker="o", label=mode)
    plt.xlabel("Seed")
    plt.ylabel("RMSE (test)")
    plt.title("Winter transfer: RMSE by seed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> None:
    args = make_parser().parse_args()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir or f"outputs_runs/{ts}_winter_transfer_multiseed")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        seed_outdir = run_one_seed(args, seed=seed, out_root=out_root)
        summary_path = seed_outdir / "summary_metrics.csv"
        if not summary_path.exists():
            raise RuntimeError(f"Missing summary for seed={seed}: {summary_path}")
        summary = pd.read_csv(summary_path)
        summary["seed"] = seed
        rows.extend(summary.to_dict(orient="records"))

    by_seed = pd.DataFrame(rows)
    by_seed = by_seed[["seed", "mode", "R2", "RMSE", "MAE", "MedAE", "n"]].sort_values(["mode", "seed"])
    by_seed.to_csv(out_root / "summary_by_seed.csv", index=False)

    agg = (
        by_seed.groupby("mode", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MedAE_mean=("MedAE", "mean"),
            MedAE_std=("MedAE", "std"),
            n_mean=("n", "mean"),
        )
        .sort_values("RMSE_mean")
    )
    agg.to_csv(out_root / "summary_agg.csv", index=False)
    plot_rmse_by_seed(by_seed, out_root / "rmse_by_seed.png")

    print(f"Saved winter multiseed run: {out_root}")


if __name__ == "__main__":
    main()
