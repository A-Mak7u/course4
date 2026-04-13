import os, datetime, optuna
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
RUN = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_seasonal")
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
    return df

def train_season(df, season_name, months):
    df_season = df[df["month"].isin(months)].copy()
    df_season = df_season[~df_season["T"].isna()]  # <=== добавил фильтр
    train = df_season[(df_season["Date"].dt.year <= 2021)]
    test  = df_season[(df_season["Date"].dt.year >= 2022)]
    val_year = int(train["Date"].dt.year.max())
    opt_train = train[train["Date"].dt.year < val_year].copy()
    opt_val = train[train["Date"].dt.year == val_year].copy()
    if train.empty or test.empty or opt_train.empty or opt_val.empty:
        raise RuntimeError(f"Пустая выборка для сезона {season_name}: train/test/opt_train/opt_val.")
    target = "T"
    features = [c for c in df_season.columns if c not in ["Cod","Date",target]]
    dtrain = xgb.DMatrix(train[features], label=train[target], missing=np.nan)
    dtest  = xgb.DMatrix(test[features],  label=test[target],  missing=np.nan)
    dtrain_opt = xgb.DMatrix(opt_train[features], label=opt_train[target], missing=np.nan)
    dval_opt = xgb.DMatrix(opt_val[features], label=opt_val[target], missing=np.nan)


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
        model = xgb.train(params, dtrain_opt, num_boost_round=2000,
                          evals=[(dval_opt,"val")],
                          early_stopping_rounds=100,
                          verbose_eval=False)
        preds = model.predict(dval_opt)
        trial.set_user_attr("best_iteration", int(getattr(model, "best_iteration", 1999)) + 1)
        return r2_score(opt_val[target], preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    })
    best_rounds = int(study.best_trial.user_attrs.get("best_iteration", 2000))

    model = xgb.train(best_params, dtrain, num_boost_round=best_rounds, verbose_eval=False)
    model.save_model(f"{OUT}/xgb_model_{season_name}.json")

    preds = model.predict(dtest)
    metrics = compute_metrics(test[target], preds)
    with open(f"{OUT}/metrics_{season_name}.txt","w") as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

df = pd.read_csv(DATA_PATH)
df = add_features(df)

train_season(df, "cold", [11,12,1,2,3])
train_season(df, "warm", [4,5,6,7,8,9,10])
print("DONE", OUT)
