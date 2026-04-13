import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import joblib
import os
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
df = pd.read_csv(DATA_PATH)

for col in ["Temperature_2m", "Dewpoint_2m"]:
    if col in df.columns:
        df[col] = df[col] - 273.15

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year
sort_cols = ["Date"] + (["Cod"] if "Cod" in df.columns else [])
df = df.sort_values(sort_cols).reset_index(drop=True)

target = "T"
features = [c for c in df.columns if c not in ["Cod", "Date", "year", "T"]]

df = df.dropna(subset=[target])
X = df[features].fillna(-999)
y = df[target]

train_mask = df["year"] <= 2021
test_mask = df["year"] >= 2022


X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("Train size:", X_train.shape, "Test size:", X_test.shape)

RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(f"outputs_runs/{RUN}", exist_ok=True)

# === Метрики ===
def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100,
        "MedAE": np.median(np.abs(y_true - y_pred)),
    }

def objective(trial):
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "objective": "reg:squarederror",

        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    best_iters = []

    for train_idx, val_idx in tscv.split(X_train):
        dtrain = xgb.DMatrix(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = xgb.DMatrix(X_train.iloc[val_idx], label=y_train.iloc[val_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=[(dval, "val")],
            early_stopping_rounds=500,
            verbose_eval=False
        )

        preds = model.predict(dval)
        scores.append(r2_score(y_train.iloc[val_idx], preds))
        best_iters.append(int(getattr(model, "best_iteration", 9999)) + 1)

    if best_iters:
        trial.set_user_attr("best_iteration", int(np.median(best_iters)))
    return np.mean(scores)

n_trials = 500  # на ночь

print(f"Запускаем {n_trials} испытаний Optuna...\n")

with tqdm(total=n_trials) as pbar:
    def callback(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

best_params = study.best_params
print("\nЛучшие параметры:", best_params)

with open(f"outputs_runs/{RUN}/best_params.txt", "w") as f:
    f.write(str(best_params) + "\n")

dtrain_full = xgb.DMatrix(X_train, label=y_train)
best_rounds = int(study.best_trial.user_attrs.get("best_iteration", 1000))
final_model = xgb.train(
    {**best_params, "tree_method": "hist", "device": "cuda", "eval_metric": "rmse"},
    dtrain_full,
    num_boost_round=best_rounds
)

joblib.dump(final_model, f"outputs_runs/{RUN}/xgb_model.pkl")

dtest = xgb.DMatrix(X_test, label=y_test)
preds = final_model.predict(dtest)
metrics = compute_metrics(y_test, preds)

with open(f"outputs_runs/{RUN}/test_metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print("\n📊 Метрики на тесте (2024–2025):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print(f"\n✅ Результаты сохранены в outputs_runs/{RUN}/")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=preds, s=10, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", lw=2)
plt.xlabel("Факт (T, °C)")
plt.ylabel("Прогноз (T, °C)")
plt.title("Фактические vs Прогноз (Test 2024–2025)")
plt.tight_layout()
plt.savefig(f"outputs_runs/{RUN}/scatter_pred_vs_true.png")
plt.close()

plt.figure(figsize=(6,4))
residuals = y_test - preds
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Ошибка (°C)")
plt.ylabel("Частота")
plt.title("Распределение ошибок (Residuals)")
plt.tight_layout()
plt.savefig(f"outputs_runs/{RUN}/residuals_hist.png")
plt.close()

plt.figure(figsize=(6,4))
sns.kdeplot(y_test, label="Факт", lw=2)
sns.kdeplot(preds, label="Прогноз", lw=2)
plt.xlabel("Температура (°C)")
plt.title("Плотности распределений")
plt.legend()
plt.tight_layout()
plt.savefig(f"outputs_runs/{RUN}/density_true_vs_pred.png")
plt.close()

print(f"📊 Графики сохранены в outputs_runs/{RUN}/")
