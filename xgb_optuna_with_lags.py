# мjдификация extra_v2: добавлены лаги t-1 для Temperature_2m, Dewpoint_2m, LST_Day, LST_Night; остальное как в extra_v2

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os, json, datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

df = pd.read_csv("final_2013_2023_T_ERA5_LST_daynight.csv")

date_cols = [c for c in ["date","Date","datetime","dt","timestamp","time"] if c in df.columns]
if date_cols:
    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
    dcol = date_cols[0]
else:
    raise RuntimeError("Не найден столбец даты (ожидались: date/Date/datetime/dt/timestamp/time)")

st_cols = [c for c in ["station_id","station","Cod","code","station_code","stationid"] if c in df.columns]
if st_cols:
    scol = st_cols[0]
else:
    df["__station__"] = 0
    scol = "__station__"

if "year" not in df.columns:
    df["year"] = df[dcol].dt.year
if "month" not in df.columns:
    df["month"] = df[dcol].dt.month
if "dayofyear" not in df.columns:
    df["dayofyear"] = df[dcol].dt.dayofyear
if "sin_doy" not in df.columns:
    df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/366.0)
if "cos_doy" not in df.columns:
    df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/366.0)
if "dewpoint_dep" not in df.columns and {"Temperature_2m","Dewpoint_2m"}.issubset(df.columns):
    df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
if "diurnal_range" not in df.columns and {"LST_Day","LST_Night"}.issubset(df.columns):
    df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]

for col in ["Temperature_2m","Dewpoint_2m","LST_Day","LST_Night"]:
    if col in df.columns:
        df = df.sort_values([scol, dcol])
        df[f"{col}_lag1"] = df.groupby(scol)[col].shift(1)

candidate_features = [
    "Temperature_2m","Dewpoint_2m","Surface_pressure","Evaporation","Total_precipitation",
    "LST_Day","LST_Night",
    "dayofyear","sin_doy","cos_doy","dewpoint_dep","diurnal_range",
    "year","month",
    "Temperature_2m_lag1","Dewpoint_2m_lag1","LST_Day_lag1","LST_Night_lag1"
]
features = [f for f in candidate_features if f in df.columns]
target = "T"
if target not in df.columns:
    raise RuntimeError("В данных нет столбца целевой переменной T")

train = df[(df["year"] >= 2013) & (df["year"] <= 2021)].copy()
test  = df[(df["year"] >= 2022) & (df["year"] <= 2023)].copy()

train = train.dropna(subset=[target])
test = test.dropna(subset=[target])

val_year = int(train["year"].max())
inner_train = train[train["year"] < val_year]
inner_val = train[train["year"] == val_year]

X_inner_train, y_inner_train = inner_train[features], inner_train[target]
X_inner_val, y_inner_val = inner_val[features], inner_val[target]
X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

d_inner_train = xgb.DMatrix(X_inner_train, label=y_inner_train)
d_inner_val = xgb.DMatrix(X_inner_val, label=y_inner_val)
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "seed": 42,
    }
    model = xgb.train(params, d_inner_train, num_boost_round=4000, evals=[(d_inner_val,"val")], early_stopping_rounds=100, verbose_eval=False)
    pred = model.predict(d_inner_val)
    r2 = r2_score(y_inner_val, pred)
    return -r2

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=60)

best_params = study.best_params
best_params.update({
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
})

model = xgb.train(best_params, d_train, num_boost_round=4000, evals=[(d_inner_val,"val")], early_stopping_rounds=100, verbose_eval=False)

pred_test = model.predict(d_test)
metrics_test = {
    "R2": float(r2_score(y_test, pred_test)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    "MAE": float(mean_absolute_error(y_test, pred_test)),
    "MedAE": float(median_absolute_error(y_test, pred_test)),
    "n_test": int(len(y_test))
}

df_full = df.dropna(subset=[target]).copy()
X_full, y_full = df_full[features], df_full[target]
d_full = xgb.DMatrix(X_full, label=y_full)
pred_full = model.predict(d_full)
metrics_full = {
    "R2": float(r2_score(y_full, pred_full)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    "MAE": float(mean_absolute_error(y_full, pred_full)),
    "MedAE": float(median_absolute_error(y_full, pred_full)),
    "n_full": int(len(y_full))
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"outputs_runs/{ts}_lags"
os.makedirs(outdir, exist_ok=True)

with open(os.path.join(outdir, "metrics_test.json"), "w") as f:
    json.dump(metrics_test, f, indent=2, ensure_ascii=False)
with open(os.path.join(outdir, "metrics_full.json"), "w") as f:
    json.dump(metrics_full, f, indent=2, ensure_ascii=False)
with open(os.path.join(outdir, "params.json"), "w") as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)
with open(os.path.join(outdir, "features_used.json"), "w") as f:
    json.dump(features, f, indent=2, ensure_ascii=False)

model.save_model(os.path.join(outdir, "model.json"))
