#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY="./.venv_geo/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python env at $PY" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="outputs_runs/${TS}_pre_rosgidromet_serious_batch"
mkdir -p "$BATCH_DIR"

LOG_MAIN="$BATCH_DIR/batch.log"
echo "[batch] start ts=$TS" | tee "$LOG_MAIN"

LOSO_OUT="outputs_runs/${TS}_saratov_loso_full"
echo "[batch] stage=1 loso out=$LOSO_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/saratov_loso_stress.py \
  --input-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --params-json outputs_runs/20250916_171729_lags123_spatial/params.json \
  --device cuda \
  --num-boost-round 2500 \
  --early-stopping-rounds 150 \
  --min-test-rows 120 \
  --seed 42 \
  --output-dir "$LOSO_OUT" \
  > "$BATCH_DIR/01_loso.log" 2>&1
echo "[batch] stage=1 done" | tee -a "$LOG_MAIN"

WINTER_OUT="outputs_runs/${TS}_volgograd_winter_multiseed_full"
echo "[batch] stage=2 winter_multiseed out=$WINTER_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/run_winter_transfer_multiseed.py \
  --source-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --target-csv data/volgograd/processed/volgograd_final_2013_2023_T_ERA5_LST_daynight.csv \
  --modes zero-shot finetune scratch \
  --seeds 42 52 62 72 82 \
  --device cuda \
  --n-trials 20 \
  --num-boost-round 3000 \
  --early-stopping-rounds 150 \
  --output-dir "$WINTER_OUT" \
  > "$BATCH_DIR/02_winter_multiseed.log" 2>&1
echo "[batch] stage=2 done" | tee -a "$LOG_MAIN"

UNC_OUT="outputs_runs/${TS}_saratov_uncertainty_full"
echo "[batch] stage=3 uncertainty out=$UNC_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/saratov_uncertainty_intervals.py \
  --input-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --device cuda \
  --n-trials 30 \
  --num-boost-round 3000 \
  --early-stopping-rounds 150 \
  --target-coverage 0.80 \
  --seed 42 \
  --output-dir "$UNC_OUT" \
  > "$BATCH_DIR/03_uncertainty.log" 2>&1
echo "[batch] stage=3 done" | tee -a "$LOG_MAIN"

echo "[batch] done ts=$TS" | tee -a "$LOG_MAIN"
echo "[batch] outputs:" | tee -a "$LOG_MAIN"
echo "  - $LOSO_OUT" | tee -a "$LOG_MAIN"
echo "  - $WINTER_OUT" | tee -a "$LOG_MAIN"
echo "  - $UNC_OUT" | tee -a "$LOG_MAIN"
