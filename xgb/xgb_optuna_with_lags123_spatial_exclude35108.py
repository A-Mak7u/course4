# XGBoost с лагами + spatial-фичи; сравнение с/без станции 35108

import os, json, datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

df = pd.read_csv("final_2013_2023_T_ERA5_LST_daynight.csv")

dcol = next(c for c in ["date","Date","datetime","dt","timestamp","time"] if c in df.columns)
df[dcol] = pd.to_datetime(df[dcol])
scol = next((c for c in ["station_id","station","Cod","code","station_code","stationid"] if c in df.columns), None)
if scol is None:
    scol = "__station__"; df[scol] = 0

if "year" not in df.columns: df["year"] = df[dcol].dt.year
if "month" not in df.columns: df["month"] = df[dcol].dt.month
if "dayofyear" not in df.columns: df["dayofyear"] = df[dcol].dt.dayofyear
if "sin_doy" not in df.columns: df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/366.0)
if "cos_doy" not in df.columns: df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/366.0)
if "dewpoint_dep" not in df.columns and {"Temperature_2m","Dewpoint_2m"}.issubset(df.columns):
    df["dewpoint_dep"] = df["Temperature_2m"] - df["Dewpoint_2m"]
if "diurnal_range" not in df.columns and {"LST_Day","LST_Night"}.issubset(df.columns):
    df["diurnal_range"] = df["LST_Day"] - df["LST_Night"]

latc = next((c for c in ["lat","latitude","Latitude","LAT","Y_final","y","Y"] if c in df.columns), None)
lonc = next((c for c in ["lon","longitude","Longitude","LON","X_final","x","X"] if c in df.columns), None)

def add_spatial(d):
    if latc and lonc:
        latv = d[latc].astype(float); lonv = d[lonc].astype(float)
        plausible_deg = (latv.abs().max()<=90 and lonv.abs().max()<=180)
        if plausible_deg:
            latr = np.deg2rad(latv); lonr = np.deg2rad(lonv)
            d["sin_lat"] = np.sin(latr); d["cos_lat"] = np.cos(latr)
            d["sin_lon"] = np.sin(lonr); d["cos_lon"] = np.cos(lonr)
        else:
            x = (lonv - lonv.min())/(lonv.max()-lonv.min()+1e-9)
            y = (latv - latv.min())/(latv.max()-latv.min()+1e-9)
            d["sin_lat"] = np.sin(2*np.pi*y); d["cos_lat"] = np.cos(2*np.pi*y)
            d["sin_lon"] = np.sin(2*np.pi*x); d["cos_lon"] = np.cos(2*np.pi*x)
    else:
        d["sin_lat"]=0.0; d["cos_lat"]=1.0; d["sin_lon"]=0.0; d["cos_lon"]=1.0
    return d

df = add_spatial(df)

df = df.sort_values([scol, dcol])
for col in ["Temperature_2m","Dewpoint_2m","LST_Day","LST_Night"]:
    if col in df.columns:
        for L in (1,2,3):
            df[f"{col}_lag{L}"] = df.groupby(scol)[col].shift(L)

target = "T"
if target not in df.columns: raise RuntimeError("нет столбца T")

base = [
    "Temperature_2m","Dewpoint_2m","Surface_pressure","Evaporation","Total_precipitation",
    "LST_Day","LST_Night","dayofyear","sin_doy","cos_doy","dewpoint_dep","diurnal_range",
    "year","month","sin_lat","cos_lat","sin_lon","cos_lon","station_train_mean_T"
]
lags = [f"{c}_lag{L}" for c in ["Temperature_2m","Dewpoint_2m","LST_Day","LST_Night"] for L in (1,2,3)]

def run_pipeline(df_in, tag):
    train = df_in[(df_in["year"]>=2013)&(df_in["year"]<=2021)].copy()
    test  = df_in[(df_in["year"]>=2022)&(df_in["year"]<=2023)].copy()

    train_mean_T = train.dropna(subset=[target]).groupby(scol)[target].mean().rename("station_train_mean_T")
    dfx = df_in.merge(train_mean_T, left_on=scol, right_index=True, how="left")
    dfx["station_train_mean_T"] = dfx["station_train_mean_T"].fillna(dfx["station_train_mean_T"].mean())

    features = [f for f in base+lags if f in dfx.columns]

    train = dfx[(dfx["year"]>=2013)&(dfx["year"]<=2021)].dropna(subset=[target]).copy()
    test  = dfx[(dfx["year"]>=2022)&(dfx["year"]<=2023)].dropna(subset=[target]).copy()
    val_year = int(train["year"].max())
    inner_train = train[train["year"]<val_year]
    inner_val   = train[train["year"]==val_year]

    def D(X, y): return xgb.DMatrix(X[features], label=y)

    study = optuna.create_study(direction="maximize")
    def objective(trial):
        p = dict(objective="reg:squarederror", tree_method="hist", device="cuda",
                 max_depth=trial.suggest_int("max_depth",4,12),
                 learning_rate=trial.suggest_float("learning_rate",0.005,0.1,log=True),
                 subsample=trial.suggest_float("subsample",0.6,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
                 reg_lambda=trial.suggest_float("reg_lambda",1e-3,10.0,log=True),
                 alpha=trial.suggest_float("alpha",1e-3,10.0,log=True),
                 min_child_weight=trial.suggest_int("min_child_weight",1,20),
                 seed=42)
        m = xgb.train(p, D(inner_train, inner_train[target]), num_boost_round=20000,
                      evals=[(D(inner_val, inner_val[target]),"val")],
                      early_stopping_rounds=300, verbose_eval=False)
        pred = m.predict(D(inner_val, inner_val[target]))
        trial.set_user_attr("best_iteration", int(getattr(m, "best_iteration", 19999)) + 1)
        return r2_score(inner_val[target], pred)

    for _ in tqdm(range(40), desc=f"Optuna {tag}"):
        study.optimize(objective, n_trials=1, catch=(Exception,))

    bp = study.best_params
    bp.update(dict(objective="reg:squarederror", tree_method="hist", device="cuda", seed=42))
    best_rounds = int(study.best_trial.user_attrs.get("best_iteration", 20000))
    model = xgb.train(bp, D(train, train[target]), num_boost_round=best_rounds, verbose_eval=False)

    def pack(y, p):
        return dict(R2=float(r2_score(y,p)),
                    RMSE=float(np.sqrt(mean_squared_error(y,p))),
                    MAE=float(mean_absolute_error(y,p)),
                    MedAE=float(median_absolute_error(y,p)),
                    n=int(len(y)))

    pred_train = model.predict(D(train, train[target]))
    pred_test  = model.predict(D(test,  test[target]))
    full_df = dfx.dropna(subset=[target]).copy()
    pred_full = model.predict(D(full_df, full_df[target]))

    metrics_train = pack(train[target], pred_train)
    metrics_test  = pack(test[target],  pred_test)
    metrics_full  = pack(full_df[target], pred_full)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"outputs_runs/{ts}_{tag}"
    os.makedirs(outdir, exist_ok=True)

    json.dump(metrics_train, open(os.path.join(outdir,"metrics_train.json"),"w"), indent=2, ensure_ascii=False)
    json.dump(metrics_test,  open(os.path.join(outdir,"metrics_test.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(metrics_full,  open(os.path.join(outdir,"metrics_full.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(bp,            open(os.path.join(outdir,"params.json"),       "w"), indent=2, ensure_ascii=False)
    json.dump(features,      open(os.path.join(outdir,"features_used.json"),"w"), indent=2, ensure_ascii=False)
    model.save_model(os.path.join(outdir,"model.json"))

    return metrics_train, metrics_test, metrics_full, outdir

print(">>> Прогон №1: со всеми станциями")
run_pipeline(df, "with35108")

print(">>> Прогон №2: без 35108")
df_wo = df[df[scol]!=35108].copy()
run_pipeline(df_wo, "without35108")
