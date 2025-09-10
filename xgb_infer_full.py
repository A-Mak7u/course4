import os, datetime, math
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

DATA_PATH  = "final_2013_2023_T_ERA5_LST_daynight.csv"
MODEL_PATH = "outputs_runs/20250905_142927/xgb_model.pkl"

RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_infer")
OUT = f"outputs_runs/{RUN}"
os.makedirs(OUT, exist_ok=True)

def smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return 100 * np.median(2.0 * np.abs(y_pred - y_true) / denom)

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, MedAE=np.nan, SMAPE=np.nan)
    return dict(
        R2   = r2_score(yt, yp),
        RMSE = math.sqrt(mean_squared_error(yt, yp)),
        MAE  = mean_absolute_error(yt, yp),
        MedAE= float(np.median(np.abs(yt - yp))),
        SMAPE= smape(yt, yp),
    )

df = pd.read_csv(DATA_PATH)
for col in ["Temperature_2m", "Dewpoint_2m"]:
    if col in df.columns:
        df[col] = df[col] - 273.15

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
else:
    df["year"] = np.nan
    df["month"] = np.nan

target = "T"
features = [c for c in df.columns if c not in ["Cod", "Date", "year", "month", "T"]]
X_all = df[features].fillna(-999)
y_all = df[target].to_numpy() if target in df.columns else np.full(len(df), np.nan)

print(f"Загружаем модель: {MODEL_PATH}")
if MODEL_PATH.endswith(".json") or MODEL_PATH.endswith(".ubj") or MODEL_PATH.endswith(".model"):
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
else:
    booster = joblib.load(MODEL_PATH)

try:
    booster.set_param({"device": "cuda"})
except Exception:
    pass

BATCH = 200_000
preds = np.zeros(len(df), dtype=np.float32)

for start in tqdm(range(0, len(df), BATCH), desc="Predict (GPU)"):
    end = min(start + BATCH, len(df))
    dmat = xgb.DMatrix(X_all.iloc[start:end])
    preds[start:end] = booster.predict(dmat)

out = pd.DataFrame({
    "Cod": df.get("Cod"),
    "Date": df.get("Date"),
    "year": df.get("year"),
    "month": df.get("month"),
    "y_true": y_all,
    "y_pred": preds,
})
out["error"] = out["y_pred"] - out["y_true"]
out["abs_error"] = np.abs(out["error"])

out.to_csv(f"{OUT}/predictions.csv", index=False)

m_overall = compute_metrics(out["y_true"].to_numpy(), out["y_pred"].to_numpy())
with open(f"{OUT}/metrics_overall.txt", "w") as f:
    for k, v in m_overall.items():
        f.write(f"{k}: {v:.4f}\n")

have_y = out[~out["y_true"].isna()].copy()

def agg_metrics(g):
    return pd.Series(compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy()))

by_year  = have_y.groupby("year", dropna=False).apply(agg_metrics).reset_index()
by_month = have_y.groupby("month", dropna=False).apply(agg_metrics).reset_index()
by_station = have_y.groupby("Cod", dropna=False).apply(agg_metrics).reset_index()

by_year.to_csv(f"{OUT}/metrics_by_year.csv", index=False)
by_month.to_csv(f"{OUT}/metrics_by_month.csv", index=False)
by_station.sort_values("RMSE", ascending=False).to_csv(f"{OUT}/metrics_by_station.csv", index=False)

print("\n=== DONE ===")
print(f"Saved: {OUT}/predictions.csv")
print(f"Saved: {OUT}/metrics_overall.txt")
print(f"Saved: {OUT}/metrics_by_year.csv")
print(f"Saved: {OUT}/metrics_by_month.csv")
print(f"Saved: {OUT}/metrics_by_station.csv")
print("\nOverall metrics:", m_overall)
