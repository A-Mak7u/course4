# xgb_infer_full_extra.py
import os, datetime, json
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
MODEL_PATH = "outputs_runs/20250914_214644_extra/xgb_model.json"  # подставь свой путь
RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_infer_extra_fix")
OUT = f"outputs_runs/{RUN}"
os.makedirs(OUT, exist_ok=True)

def smape(y_true, y_pred, eps=1e-6):
    d = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return 100 * float(np.median(2.0 * np.abs(y_pred - y_true) / d))

def compute_metrics(y_true, y_pred):
    m = ~np.isnan(y_true)
    yt, yp = y_true[m], y_pred[m]
    return dict(
        R2=float(r2_score(yt, yp)),
        RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
        MAE=float(mean_absolute_error(yt, yp)),
        MedAE=float(np.median(np.abs(yt - yp))),
        SMAPE=smape(yt, yp),
    )

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear
df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365)
df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365)
df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]

target = "T"
features = [c for c in df.columns if c not in ["Cod","Date","year",target]]

X = df[features]                   # ВАЖНО: без fillna
y = df[target].to_numpy() if target in df.columns else np.full(len(df), np.nan)

booster = xgb.Booster()
booster.load_model(MODEL_PATH)
try: booster.set_param({"device":"cuda"})
except: pass

BATCH = 200_000
preds = np.zeros(len(df), dtype=np.float32)
for s in tqdm(range(0, len(df), BATCH), desc="Predict (GPU)"):
    e = min(s+BATCH, len(df))
    dmat = xgb.DMatrix(X.iloc[s:e], missing=np.nan)
    preds[s:e] = booster.predict(dmat)

out = pd.DataFrame({
    "Cod": df.get("Cod"),
    "Date": df.get("Date"),
    "year": df.get("year"),
    "month": df.get("month"),
    "y_true": y,
    "y_pred": preds,
})
out["error"] = out["y_pred"] - out["y_true"]
out["abs_error"] = np.abs(out["error"])
out.to_csv(f"{OUT}/predictions.csv", index=False)

m_overall = compute_metrics(out["y_true"].to_numpy(), out["y_pred"].to_numpy())
with open(f"{OUT}/metrics_overall.json","w") as f: json.dump(m_overall, f, indent=2)

have = out[~out["y_true"].isna()].copy()
def agg(g):
    d = compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    return pd.Series(d)
by_year = have.groupby("year", dropna=False).apply(agg).reset_index()
by_month = have.groupby("month", dropna=False).apply(agg).reset_index()
by_station = have.groupby("Cod", dropna=False).apply(agg).reset_index()
by_year.to_csv(f"{OUT}/metrics_by_year.csv", index=False)
by_month.to_csv(f"{OUT}/metrics_by_month.csv", index=False)
by_station.sort_values("RMSE", ascending=False).to_csv(f"{OUT}/metrics_by_station.csv", index=False)

plt.figure(figsize=(6,6))
plt.scatter(have["y_true"], have["y_pred"], s=6, alpha=0.25)
mn, mx = have["y_true"].min(), have["y_true"].max()
plt.plot([mn,mx],[mn,mx],"r--")
plt.xlabel("Факт"); plt.ylabel("Прогноз"); plt.title("Scatter: факт vs прогноз (весь датасет)")
plt.savefig(f"{OUT}/scatter_pred_vs_true.png", dpi=150); plt.close()

res = have["y_true"] - have["y_pred"]
plt.figure(figsize=(6,4))
plt.hist(res, bins=60, alpha=0.8)
plt.xlabel("Ошибка (факт - прогноз)"); plt.ylabel("Частота"); plt.title("Histogram остатков")
plt.savefig(f"{OUT}/residuals_hist.png", dpi=150); plt.close()

plt.figure(figsize=(6,4))
plt.hist(have["y_true"], bins=60, alpha=0.5, label="Факт")
plt.hist(have["y_pred"], bins=60, alpha=0.5, label="Прогноз")
plt.legend(); plt.title("Распределение: факт vs прогноз")
plt.savefig(f"{OUT}/density_true_vs_pred.png", dpi=150); plt.close()

print("DONE", OUT, m_overall)
