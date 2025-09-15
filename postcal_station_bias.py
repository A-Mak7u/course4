import os, datetime
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt

RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_biascorr")
OUT = f"outputs_runs/{RUN}"
os.makedirs(OUT, exist_ok=True)

INPUT = "outputs_runs/20250914_220035_infer_extra_fix/predictions.csv"  # путь к predictions extra модели

def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }

df = pd.read_csv(INPUT, parse_dates=["Date"])

train = df[(df["Date"].dt.year <= 2021) & (~df["y_true"].isna())]
bias = train.groupby("Cod").apply(lambda g: (g["y_pred"] - g["y_true"]).mean())
df = df.merge(bias.rename("station_bias"), on="Cod", how="left")
df["y_pred_corr"] = df["y_pred"] - df["station_bias"].fillna(0)

mask = ~df["y_true"].isna()
m_before = compute_metrics(df.loc[mask, "y_true"], df.loc[mask, "y_pred"])
m_after  = compute_metrics(df.loc[mask, "y_true"], df.loc[mask, "y_pred_corr"])

with open(f"{OUT}/metrics_biascorr.txt","w") as f:
    f.write("=== BEFORE ===\n")
    for k,v in m_before.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("\n=== AFTER ===\n")
    for k,v in m_after.items():
        f.write(f"{k}: {v:.4f}\n")

by_station = df.loc[mask].groupby("Cod").apply(
    lambda g: pd.Series(compute_metrics(g["y_true"], g["y_pred_corr"]))
).reset_index()
by_station.to_csv(f"{OUT}/metrics_by_station_corr.csv", index=False)

plt.figure(figsize=(6,6))
plt.scatter(df.loc[mask, "y_true"], df.loc[mask, "y_pred"], s=4, alpha=0.3, label="До коррекции")
plt.scatter(df.loc[mask, "y_true"], df.loc[mask, "y_pred_corr"], s=4, alpha=0.3, label="После коррекции")
mn,mx = df["y_true"].min(), df["y_true"].max()
plt.plot([mn,mx],[mn,mx],"r--")
plt.legend(); plt.xlabel("Факт"); plt.ylabel("Прогноз")
plt.title("Scatter: до и после коррекции по станциям")
plt.savefig(f"{OUT}/scatter_biascorr.png", dpi=150)
plt.close()

print("Готово:", OUT)
print("Метрики до:", m_before)
print("Метрики после:", m_after)
