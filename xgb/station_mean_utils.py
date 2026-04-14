from __future__ import annotations

import pandas as pd


def apply_station_train_mean_feature(
    df_target: pd.DataFrame,
    df_reference_train: pd.DataFrame,
    station_col: str,
    target_col: str = "T",
    feature_name: str = "station_train_mean_T",
) -> pd.DataFrame:
    out = df_target.copy()
    ref = df_reference_train.dropna(subset=[target_col]).copy()
    station_mean = ref.groupby(station_col)[target_col].mean().rename(feature_name)
    global_fill = float(ref[target_col].mean()) if not ref.empty else 0.0
    out = out.merge(station_mean, left_on=station_col, right_index=True, how="left")
    out[feature_name] = out[feature_name].fillna(global_fill)
    return out
