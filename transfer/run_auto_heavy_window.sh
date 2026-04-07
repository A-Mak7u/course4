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
BATCH_DIR="outputs_runs/${TS}_auto_heavy_window"
mkdir -p "$BATCH_DIR"
LOG_MAIN="$BATCH_DIR/batch.log"

echo "[batch] ts=$TS" | tee -a "$LOG_MAIN"
echo "[batch] root=$ROOT_DIR" | tee -a "$LOG_MAIN"
echo "[batch] python=$PY" | tee -a "$LOG_MAIN"

WINTER_OUT="outputs_runs/${TS}_volgograd_winter_multiseed_x10"
echo "[batch] stage=1 winter_multiseed_x10 out=$WINTER_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/run_winter_transfer_multiseed.py \
  --source-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --target-csv data/volgograd/processed/volgograd_final_2013_2023_T_ERA5_LST_daynight.csv \
  --modes zero-shot finetune scratch \
  --seeds 42 52 62 72 82 92 102 112 122 132 \
  --device cuda \
  --n-trials 25 \
  --num-boost-round 3500 \
  --early-stopping-rounds 150 \
  --output-dir "$WINTER_OUT" \
  > "$BATCH_DIR/01_winter_multiseed_x10.log" 2>&1
echo "[batch] stage=1 done" | tee -a "$LOG_MAIN"

SPATIAL_OUT="outputs_runs/${TS}_spatial_transfer_preflight_serious"
echo "[batch] stage=2 spatial_preflight_serious out=$SPATIAL_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/spatial_transfer_preflight.py \
  --input-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --station-col Cod \
  --date-col Date \
  --target-col T \
  --lon-col X_final \
  --lat-col Y_final \
  --target-max-stations-grid 0 5 3 \
  --modes zero-shot finetune scratch \
  --device cuda \
  --n-trials 18 \
  --num-boost-round 3000 \
  --early-stopping-rounds 150 \
  --seed 42 \
  --zero-inflated-precip \
  --output-dir "$SPATIAL_OUT" \
  > "$BATCH_DIR/02_spatial_preflight_serious.log" 2>&1
echo "[batch] stage=2 done" | tee -a "$LOG_MAIN"

UNC80_OUT="outputs_runs/${TS}_saratov_uncertainty_cov80_strict"
echo "[batch] stage=3 uncertainty_cov80_strict out=$UNC80_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/saratov_uncertainty_intervals.py \
  --input-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --device cuda \
  --n-trials 40 \
  --num-boost-round 3500 \
  --early-stopping-rounds 150 \
  --target-coverage 0.80 \
  --seed 42 \
  --output-dir "$UNC80_OUT" \
  > "$BATCH_DIR/03_uncertainty_cov80_strict.log" 2>&1
echo "[batch] stage=3 done" | tee -a "$LOG_MAIN"

UNC85_OUT="outputs_runs/${TS}_saratov_uncertainty_cov85_strict"
echo "[batch] stage=4 uncertainty_cov85_strict out=$UNC85_OUT" | tee -a "$LOG_MAIN"
"$PY" transfer/saratov_uncertainty_intervals.py \
  --input-csv final_2013_2023_T_ERA5_LST_daynight.csv \
  --device cuda \
  --n-trials 40 \
  --num-boost-round 3500 \
  --early-stopping-rounds 150 \
  --target-coverage 0.85 \
  --seed 42 \
  --output-dir "$UNC85_OUT" \
  > "$BATCH_DIR/04_uncertainty_cov85_strict.log" 2>&1
echo "[batch] stage=4 done" | tee -a "$LOG_MAIN"

echo "[batch] finished" | tee -a "$LOG_MAIN"
echo "[batch] outputs:" | tee -a "$LOG_MAIN"
echo "  - $WINTER_OUT" | tee -a "$LOG_MAIN"
echo "  - $SPATIAL_OUT" | tee -a "$LOG_MAIN"
echo "  - $UNC80_OUT" | tee -a "$LOG_MAIN"
echo "  - $UNC85_OUT" | tee -a "$LOG_MAIN"
