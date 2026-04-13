
import os, datetime, json
import argparse
import pandas as pd, numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_infer_extra_v2")
OUT = f"outputs_runs/{RUN}"
os.makedirs(OUT, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference extra_v2 на полном датасете без переобучения."
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Путь к xgb_model.json. Если не задан, берётся последний outputs_runs/*_extra_v2/xgb_model.json",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="CSV с входными данными.",
    )
    return p.parse_args()


def resolve_model_path(model_path_arg: str) -> Path:
    if model_path_arg:
        p = Path(model_path_arg)
        if not p.exists():
            raise FileNotFoundError(f"Модель не найдена: {p}")
        return p

    candidates = sorted(Path("outputs_runs").glob("*_extra_v2/xgb_model.json"))
    if not candidates:
        raise FileNotFoundError("Не найдено ни одной модели outputs_runs/*_extra_v2/xgb_model.json")
    return candidates[-1]

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365)
    df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365)
    df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
    df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]
    df["has_LST_Day"] = (~df["LST_Day"].isna()).astype(int)
    df["has_LST_Night"] = (~df["LST_Night"].isna()).astype(int)
    return df

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MedAE": float(np.median(np.abs(yt - yp))),
    }

args = parse_args()
model_path = resolve_model_path(args.model_path)
df = pd.read_csv(args.data_path)
df = add_features(df)

target = "T"
features = [c for c in df.columns if c not in ["Cod","Date","year",target]]

booster = xgb.Booster()
booster.load_model(str(model_path))
booster.set_param({"device":"cuda"})

dmat = xgb.DMatrix(df[features], missing=np.nan)
preds = booster.predict(dmat)

df["y_pred"] = preds
df["error"] = df["y_pred"] - df[target]
df["abs_error"] = np.abs(df["error"])
df.to_csv(f"{OUT}/predictions.csv", index=False)

# === Общие метрики ===
metrics = compute_metrics(df[target].to_numpy(), df["y_pred"].to_numpy())
with open(f"{OUT}/metrics_overall.json","w") as f:
    json.dump(metrics, f, indent=2)

# === Разбивки ===
def agg_metrics(g):
    mask = ~g[target].isna()
    if mask.sum() == 0:
        return pd.Series({"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MedAE": np.nan})
    return pd.Series(compute_metrics(g.loc[mask, target].to_numpy(),
                                     g.loc[mask, "y_pred"].to_numpy()))

df.groupby("year").apply(agg_metrics).reset_index().to_csv(f"{OUT}/metrics_by_year.csv", index=False)
df.groupby("month").apply(agg_metrics).reset_index().to_csv(f"{OUT}/metrics_by_month.csv", index=False)
if "Cod" in df.columns:
    df.groupby("Cod").apply(agg_metrics).reset_index().to_csv(f"{OUT}/metrics_by_station.csv", index=False)


# === Графики ===
# scatter
plt.figure(figsize=(6,6))
plt.scatter(df[target], df["y_pred"], s=3, alpha=0.25)
mn, mx = df[target].min(), df[target].max()
plt.plot([mn,mx],[mn,mx],"r--")
plt.xlabel("Факт (°C)"); plt.ylabel("Прогноз (°C)")
plt.title("Scatter: факт vs прогноз (extra_v2)")
plt.savefig(f"{OUT}/scatter_pred_vs_true.png", dpi=150); plt.close()

# гистограмма ошибок
plt.figure(figsize=(6,4))
plt.hist(df["error"].dropna(), bins=80, alpha=0.8)
plt.xlabel("Ошибка (°C)"); plt.ylabel("Частота")
plt.title("Histogram остатков")
plt.savefig(f"{OUT}/residuals_hist.png", dpi=150); plt.close()

# сравнение распределений
plt.figure(figsize=(6,4))
bins = np.linspace(min(df[target].min(), df["y_pred"].min()),
                   max(df[target].max(), df["y_pred"].max()), 70)
plt.hist(df[target].dropna(), bins=bins, alpha=0.5, label="Факт", density=True)
plt.hist(df["y_pred"], bins=bins, alpha=0.5, label="Прогноз", density=True)
plt.legend(); plt.title("Плотности распределений (extra_v2)")
plt.xlabel("Температура (°C)"); plt.ylabel("Density")
plt.savefig(f"{OUT}/density_true_vs_pred.png", dpi=150); plt.close()

# Boxplot ошибок по месяцам
plt.figure(figsize=(8,5))
df.boxplot(column="error", by="month", grid=False)
plt.title("Ошибки по месяцам"); plt.suptitle("")
plt.xlabel("Месяц"); plt.ylabel("Ошибка (°C)")
plt.savefig(f"{OUT}/boxplot_error_by_month.png", dpi=150); plt.close()

print("MODEL_PATH", str(model_path))
print("DONE", OUT, metrics)
