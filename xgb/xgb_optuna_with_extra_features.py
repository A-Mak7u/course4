import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import os, datetime, json, matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
OUTPUT_DIR = f"outputs_runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_extra"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET = "T"

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape,
        "MedAE": median_absolute_error(y_true, y_pred),
    }

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).copy()
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]

train = df[df["year"] <= 2021]
test  = df[df["year"] >= 2022]
val_year = int(train["year"].max())
opt_train = train[train["year"] < val_year].copy()
opt_val = train[train["year"] == val_year].copy()
if opt_train.empty or opt_val.empty:
    raise RuntimeError("Не удалось собрать внутренний split train/val из train-периода.")

features = [c for c in df.columns if c not in [TARGET, "Cod", "Date", "year"]]
X_train, y_train = train[features], train[TARGET]
X_test,  y_test  = test[features],  test[TARGET]
X_opt_train, y_opt_train = opt_train[features], opt_train[TARGET]
X_opt_val, y_opt_val = opt_val[features], opt_val[TARGET]
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)
dtrain_opt = xgb.DMatrix(X_opt_train, label=y_opt_train)
dval_opt = xgb.DMatrix(X_opt_val, label=y_opt_val)

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }
    booster = xgb.train(
        params,
        dtrain_opt,
        num_boost_round=500,
        evals=[(dval_opt, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    preds = booster.predict(dval_opt)
    trial.set_user_attr("best_iteration", int(getattr(booster, "best_iteration", 499)) + 1)
    return r2_score(y_opt_val, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_rounds = int(study.best_trial.user_attrs.get("best_iteration", 500))
booster = xgb.train(
    {**best_params, "objective": "reg:squarederror", "tree_method": "hist", "device": "cuda"},
    dtrain,
    num_boost_round=best_rounds,
    verbose_eval=False,
)

preds = booster.predict(dtest)
metrics = compute_metrics(y_test, preds)

print("📊 Метрики на тесте (2022–2023):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

with open(f"{OUTPUT_DIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
with open(f"{OUTPUT_DIR}/metrics_test.json", "w") as f:
    json.dump(metrics, f, indent=2)
booster.save_model(f"{OUTPUT_DIR}/xgb_model.json")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Факт")
plt.ylabel("Прогноз")
plt.title("Scatter: факт vs прогноз")
plt.savefig(f"{OUTPUT_DIR}/scatter_pred_vs_true.png", dpi=150)
plt.close()

residuals = y_test - preds
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.xlabel("Ошибка (факт - прогноз)")
plt.ylabel("Частота")
plt.title("Histogram остатков")
plt.savefig(f"{OUTPUT_DIR}/residuals_hist.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(y_test, bins=50, alpha=0.5, label="Факт")
plt.hist(preds, bins=50, alpha=0.5, label="Прогноз")
plt.legend()
plt.title("Распределение: факт vs прогноз")
plt.savefig(f"{OUTPUT_DIR}/density_true_vs_pred.png", dpi=150)
plt.close()

print("✅ Результаты и графики сохранены в:", OUTPUT_DIR)
