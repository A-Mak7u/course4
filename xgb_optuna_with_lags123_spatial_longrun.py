# модификация lags123_spatial: длинный бустинг (num_boost_round=20000, early_stopping_rounds=300, lr [0.005,0.10])

import os, json, datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("final_2013_2023_T_ERA5_LST_daynight.csv")

dcol = next(c for c in ["date","Date","datetime","dt","timestamp","time"] if c in df.columns); df[dcol] = pd.to_datetime(df[dcol])
scol = next((c for c in ["station_id","station","Cod","code","station_code","stationid"] if c in df.columns), None) or "__station__"
if scol not in df.columns: df[scol]=0

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
        if (latv.abs().max()<=90) and (lonv.abs().max()<=180):
            latr = np.deg2rad(latv); lonr = np.deg2rad(lonv)
            d["sin_lat"] = np.sin(latr); d["cos_lat"] = np.cos(latr)
            d["sin_lon"] = np.sin(lonr); d["cos_lon"] = np.cos(lonr)
        else:
            x = (lonv - lonv.min())/(lonv.max()-lonv.min() + 1e-9)
            y = (latv - latv.min())/(latv.max()-latv.min() + 1e-9)
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

if "T" not in df.columns: raise RuntimeError("нет столбца T")
train = df[(df["year"]>=2013)&(df["year"]<=2021)].copy()
test  = df[(df["year"]>=2022)&(df["year"]<=2023)].copy()

train_mean_T = train.dropna(subset=["T"]).groupby(scol)["T"].mean().rename("station_train_mean_T")
df = df.merge(train_mean_T, left_on=scol, right_index=True, how="left")
df["station_train_mean_T"] = df["station_train_mean_T"].fillna(df["station_train_mean_T"].mean())

base = ["Temperature_2m","Dewpoint_2m","Surface_pressure","Evaporation","Total_precipitation",
        "LST_Day","LST_Night","dayofyear","sin_doy","cos_doy","dewpoint_dep","diurnal_range",
        "year","month","sin_lat","cos_lat","sin_lon","cos_lon","station_train_mean_T"]
lags = [f"{c}_lag{L}" for c in ["Temperature_2m","Dewpoint_2m","LST_Day","LST_Night"] for L in (1,2,3)]
features = [f for f in base+lags if f in df.columns]

train = df[(df["year"]>=2013)&(df["year"]<=2021)].dropna(subset=["T"]).copy()
test  = df[(df["year"]>=2022)&(df["year"]<=2023)].dropna(subset=["T"]).copy()
val_year = int(train["year"].max())
inner_train = train[train["year"]<val_year]
inner_val   = train[train["year"]==val_year]

def D(X, y): return xgb.DMatrix(X[features], label=y)

optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction="maximize")
def objective(trial):
    p = dict(objective="reg:squarederror", tree_method="hist", device="cuda",
             max_depth=trial.suggest_int("max_depth",4,12),
             learning_rate=trial.suggest_float("learning_rate",0.005,0.10,log=True),
             subsample=trial.suggest_float("subsample",0.6,1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
             reg_lambda=trial.suggest_float("reg_lambda",1e-3,10.0,log=True),
             alpha=trial.suggest_float("alpha",1e-3,10.0,log=True),
             min_child_weight=trial.suggest_int("min_child_weight",1,20),
             seed=42)
    m = xgb.train(p, D(inner_train, inner_train["T"]), num_boost_round=20000,
                  evals=[(D(inner_val, inner_val["T"]),"val")],
                  early_stopping_rounds=300, verbose_eval=False)
    pred = m.predict(D(inner_val, inner_val["T"]))
    return r2_score(inner_val["T"], pred)
study.optimize(objective, n_trials=60)

bp = study.best_params
bp.update(dict(objective="reg:squarederror", tree_method="hist", device="cuda", seed=42))
model = xgb.train(bp, D(train, train["T"]), num_boost_round=20000,
                  evals=[(D(inner_val, inner_val["T"]),"val")],
                  early_stopping_rounds=300, verbose_eval=50)

def pack(y, p):
    return dict(R2=float(r2_score(y,p)),
                RMSE=float(np.sqrt(mean_squared_error(y,p))),
                MAE=float(mean_absolute_error(y,p)),
                MedAE=float(median_absolute_error(y,p)),
                n=int(len(y)))

pred_train = model.predict(D(train, train["T"]))
pred_test  = model.predict(D(test,  test["T"]))
full_df = df.dropna(subset=["T"]).copy()
pred_full = model.predict(D(full_df, full_df["T"]))

metrics_train = pack(train["T"], pred_train)
metrics_test  = pack(test["T"],  pred_test)
metrics_full  = pack(full_df["T"], pred_full)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"outputs_runs/{ts}_lags123_spatial_longrun"
os.makedirs(outdir, exist_ok=True)

json.dump(metrics_train, open(os.path.join(outdir,"metrics_train.json"),"w"), indent=2, ensure_ascii=False)
json.dump(metrics_test,  open(os.path.join(outdir,"metrics_test.json"), "w"), indent=2, ensure_ascii=False)
json.dump(metrics_full,  open(os.path.join(outdir,"metrics_full.json"), "w"), indent=2, ensure_ascii=False)
json.dump(bp,            open(os.path.join(outdir,"params.json"),       "w"), indent=2, ensure_ascii=False)
json.dump(features,      open(os.path.join(outdir,"features_used.json"),"w"), indent=2, ensure_ascii=False)
model.save_model(os.path.join(outdir,"model.json"))

def save_group_metrics(df_src, y_true, y_pred, by_col, path_csv):
    g = df_src.assign(__y=y_true, __p=y_pred).groupby(by_col)
    rows = []
    for k, s in g:
        if len(s) < 5: continue
        rows.append(dict(group=k, n=len(s),
                         R2=float(r2_score(s["__y"], s["__p"])) if len(s)>=2 else np.nan,
                         RMSE=float(np.sqrt(mean_squared_error(s["__y"], s["__p"]))),
                         MAE=float(mean_absolute_error(s["__y"], s["__p"])),
                         MedAE=float(median_absolute_error(s["__y"], s["__p"]))))
    pd.DataFrame(rows).sort_values("group").to_csv(path_csv, index=False)

save_group_metrics(test,  test["T"],  pred_test,  "month", os.path.join(outdir,"metrics_by_month_test.csv"))
save_group_metrics(test,  test["T"],  pred_test,  scol,    os.path.join(outdir,"metrics_by_station_test.csv"))
save_group_metrics(full_df, full_df["T"], pred_full, "month", os.path.join(outdir,"metrics_by_month_full.csv"))
save_group_metrics(full_df, full_df["T"], pred_full, scol,    os.path.join(outdir,"metrics_by_station_full.csv"))

try:
    plt.figure(figsize=(6,6))
    lims = [float(min(test["T"].min(), pred_test.min())), float(max(test["T"].max(), pred_test.max()))]
    plt.scatter(test["T"], pred_test, s=4, alpha=0.5); plt.plot(lims, lims)
    plt.xlabel("True (test)"); plt.ylabel("Pred (test)"); plt.title("Pred vs True (test)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"scatter_pred_vs_true.png"), dpi=160); plt.close()
except Exception: pass

try:
    plt.figure(figsize=(6,4))
    resid = pred_test - test["T"].to_numpy()
    plt.hist(resid, bins=60); plt.xlabel("Residual (test)"); plt.ylabel("Count"); plt.title("Residuals histogram (test)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"residuals_hist.png"), dpi=160); plt.close()
except Exception: pass

try:
    plt.figure(figsize=(7,4))
    box = []
    for m in range(1,13):
        s = full_df[full_df["month"]==m]
        if len(s) >= 5:
            box.append((pred_full[full_df["month"]==m] - s["T"].to_numpy()))
    plt.boxplot([b for b in box if len(b)>0], tick_labels=[str(i) for i in range(1,1+len(box))])
    plt.xlabel("Month"); plt.ylabel("Error (Pred-True)"); plt.title("Error by month (full)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"boxplot_error_by_month.png"), dpi=160); plt.close()
except Exception: pass

try:
    from scipy.stats import gaussian_kde
    plt.figure(figsize=(6,5))
    x = full_df["T"].to_numpy(); y = pred_full; xy = np.vstack([x,y])
    try:
        z = gaussian_kde(xy)(xy); idx = z.argsort(); x, y, z = x[idx], y[idx], z[idx]; plt.scatter(x, y, c=z, s=3)
    except Exception:
        plt.scatter(x, y, s=3, alpha=0.3)
    lims = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    plt.plot(lims, lims); plt.xlabel("True (full)"); plt.ylabel("Pred (full)"); plt.title("Density True vs Pred (full)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"density_true_vs_pred_full.png"), dpi=160); plt.close()
except Exception: pass
