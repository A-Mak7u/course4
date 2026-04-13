# XGBoost с lag+spatial, только для зимних месяцев (11–3)

import os, json, datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

df = pd.read_csv("final_2013_2023_T_ERA5_LST_daynight.csv")

dcol = next(c for c in ["date","Date","datetime","dt","timestamp","time"] if c in df.columns)
df[dcol] = pd.to_datetime(df[dcol])
scol = next((c for c in ["station_id","station","Cod","code","station_code","stationid"] if c in df.columns), None)
if scol is None: scol="__station__"; df[scol]=0
if "year" not in df.columns: df["year"]=df[dcol].dt.year
if "month" not in df.columns: df["month"]=df[dcol].dt.month
if "dayofyear" not in df.columns: df["dayofyear"]=df[dcol].dt.dayofyear
if "sin_doy" not in df.columns: df["sin_doy"]=np.sin(2*np.pi*df["dayofyear"]/366.0)
if "cos_doy" not in df.columns: df["cos_doy"]=np.cos(2*np.pi*df["dayofyear"]/366.0)
if "dewpoint_dep" not in df.columns and {"Temperature_2m","Dewpoint_2m"}.issubset(df.columns):
    df["dewpoint_dep"]=df["Temperature_2m"]-df["Dewpoint_2m"]
if "diurnal_range" not in df.columns and {"LST_Day","LST_Night"}.issubset(df.columns):
    df["diurnal_range"]=df["LST_Day"]-df["LST_Night"]

lat_candidates=[c for c in ["lat","latitude","Latitude","LAT","Y_final","y","Y"] if c in df.columns]
lon_candidates=[c for c in ["lon","longitude","Longitude","LON","X_final","x","X"] if c in df.columns]
latc=lat_candidates[0] if lat_candidates else None
lonc=lon_candidates[0] if lon_candidates else None

def add_spatial(d):
    if latc and lonc:
        latv=d[latc].astype(float); lonv=d[lonc].astype(float)
        plausible_deg=(latv.abs().max()<=90 and lonv.abs().max()<=180)
        if plausible_deg:
            latr=np.deg2rad(latv); lonr=np.deg2rad(lonv)
            d["sin_lat"]=np.sin(latr); d["cos_lat"]=np.cos(latr)
            d["sin_lon"]=np.sin(lonr); d["cos_lon"]=np.cos(lonr)
        else:
            x=(lonv-lonv.min())/(lonv.max()-lonv.min()+1e-9)
            y=(latv-latv.min())/(latv.max()-latv.min()+1e-9)
            d["sin_lat"]=np.sin(2*np.pi*y); d["cos_lat"]=np.cos(2*np.pi*y)
            d["sin_lon"]=np.sin(2*np.pi*x); d["cos_lon"]=np.cos(2*np.pi*x)
    else:
        d["sin_lat"]=0.0; d["cos_lat"]=1.0; d["sin_lon"]=0.0; d["cos_lon"]=1.0
    return d

df=add_spatial(df)

df=df.sort_values([scol,dcol])
for col in ["Temperature_2m","Dewpoint_2m","LST_Day","LST_Night"]:
    if col in df.columns:
        for L in (1,2,3):
            df[f"{col}_lag{L}"]=df.groupby(scol)[col].shift(L)

target="T"
if target not in df.columns: raise RuntimeError("нет столбца T")

winter_months=[11,12,1,2,3]
df=df[df["month"].isin(winter_months)].copy()

train=df[(df["year"]>=2013)&(df["year"]<=2021)].dropna(subset=[target])
test=df[(df["year"]>=2022)&(df["year"]<=2023)].dropna(subset=[target])
val_year=int(train["year"].max())
inner_train=train[train["year"]<val_year]; inner_val=train[train["year"]==val_year]

features=[c for c in df.columns if c not in [target,dcol]]
def D(X,y): return xgb.DMatrix(X[features],label=y)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study=optuna.create_study(direction="maximize")

def objective(trial):
    p=dict(objective="reg:squarederror",tree_method="hist",device="cuda",
           max_depth=trial.suggest_int("max_depth",4,12),
           learning_rate=trial.suggest_float("learning_rate",0.005,0.1,log=True),
           subsample=trial.suggest_float("subsample",0.6,1.0),
           colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
           reg_lambda=trial.suggest_float("reg_lambda",1e-3,10.0,log=True),
           alpha=trial.suggest_float("alpha",1e-3,10.0,log=True),
           min_child_weight=trial.suggest_int("min_child_weight",1,20),seed=42)
    m=xgb.train(p,D(inner_train,inner_train[target]),num_boost_round=20000,
                evals=[(D(inner_val,inner_val[target]),"val")],
                early_stopping_rounds=300,verbose_eval=False)
    pred=m.predict(D(inner_val,inner_val[target]))
    trial.set_user_attr("best_iteration", int(getattr(m, "best_iteration", 19999)) + 1)
    return r2_score(inner_val[target],pred)

for _ in tqdm(range(60),desc="Optuna trials"):
    study.optimize(objective,n_trials=1)

bp=study.best_params
bp.update(dict(objective="reg:squarederror",tree_method="hist",device="cuda",seed=42))
best_rounds = int(study.best_trial.user_attrs.get("best_iteration", 20000))
model=xgb.train(bp,D(train,train[target]),num_boost_round=best_rounds,verbose_eval=False)

def pack(y,p): return dict(R2=float(r2_score(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),
                           MAE=float(mean_absolute_error(y,p)),MedAE=float(median_absolute_error(y,p)),n=int(len(y)))

pred_train=model.predict(D(train,train[target]))
pred_test=model.predict(D(test,test[target]))
full_df=df.dropna(subset=[target])
pred_full=model.predict(D(full_df,full_df[target]))

metrics_train=pack(train[target],pred_train)
metrics_test=pack(test[target],pred_test)
metrics_full=pack(full_df[target],pred_full)

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir=f"outputs_runs/{ts}_winter_only"
os.makedirs(outdir,exist_ok=True)

json.dump(metrics_train,open(os.path.join(outdir,"metrics_train.json"),"w"),indent=2,ensure_ascii=False)
json.dump(metrics_test,open(os.path.join(outdir,"metrics_test.json"),"w"),indent=2,ensure_ascii=False)
json.dump(metrics_full,open(os.path.join(outdir,"metrics_full.json"),"w"),indent=2,ensure_ascii=False)
json.dump(bp,open(os.path.join(outdir,"params.json"),"w"),indent=2,ensure_ascii=False)
json.dump(features,open(os.path.join(outdir,"features_used.json"),"w"),indent=2,ensure_ascii=False)
model.save_model(os.path.join(outdir,"model.json"))

try:
    plt.figure(figsize=(6,6))
    lims=[float(min(test[target].min(),pred_test.min())),float(max(test[target].max(),pred_test.max()))]
    plt.scatter(test[target],pred_test,s=4,alpha=0.5)
    plt.plot(lims,lims); plt.xlabel("True (test)"); plt.ylabel("Pred (test)")
    plt.title("Pred vs True (test)"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"scatter_pred_vs_true.png"),dpi=160); plt.close()
except: pass

try:
    plt.figure(figsize=(6,4))
    resid=pred_test-test[target].to_numpy()
    plt.hist(resid,bins=60); plt.xlabel("Residual (test)"); plt.ylabel("Count")
    plt.title("Residuals histogram (test)"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"residuals_hist.png"),dpi=160); plt.close()
except: pass
