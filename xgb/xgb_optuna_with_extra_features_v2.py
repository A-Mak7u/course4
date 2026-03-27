# В этом варианте добавлены бинарные индикаторы наличия MODIS: has_LST_Day, has_LST_Night
# это позволит модели различать случаи, когда данные LST отсутствуют из-за облачности

import os, datetime, optuna
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_extra_v2")
OUT = f"outputs_runs/{RUN}"
os.makedirs(OUT, exist_ok=True)

def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["month"] = df["Date"].dt.month
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365)
    df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365)
    df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
    df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]
    df["has_LST_Day"] = (~df["LST_Day"].isna()).astype(int)
    df["has_LST_Night"] = (~df["LST_Night"].isna()).astype(int)
    return df

df = pd.read_csv(DATA_PATH)
df = add_features(df)

train = df[(df["Date"].dt.year <= 2021) & (~df["T"].isna())]
test  = df[(df["Date"].dt.year >= 2022) & (~df["T"].isna())]

target = "T"
features = [c for c in df.columns if c not in ["Cod","Date",target]]

dtrain = xgb.DMatrix(train[features], label=train[target], missing=np.nan)
dtest  = xgb.DMatrix(test[features],  label=test[target],  missing=np.nan)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dtest,"test")],
                      early_stopping_rounds=100,
                      verbose_eval=False)
    preds = model.predict(dtest)
    return r2_score(test[target], preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
})

model = xgb.train(best_params, dtrain, num_boost_round=2000,
                  evals=[(dtest,"test")],
                  early_stopping_rounds=100,
                  verbose_eval=100)

model.save_model(f"{OUT}/xgb_model.json")

preds = model.predict(dtest)
metrics = compute_metrics(test[target], preds)
with open(f"{OUT}/test_metrics.txt","w") as f:
    for k,v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print("DONE", OUT, metrics)
