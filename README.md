# Course4: Temperature Reconstruction (2013-2023)

Проект по восстановлению наземной температуры `T` на основе ERA5 + MODIS + признаков станции/времени.

## Что есть в репозитории

- `final_2013_2023_T_ERA5_LST_daynight.csv` — основной датасет (`208278` строк, `12` колонок).
- `eda.py` и `eda_plots/` — разведочный анализ и графики.
- Скрипты серии `xgb_*` — эксперименты с XGBoost (база, extra-features, lags, spatial, diagnostics).
- `outputs_runs/` — сохранённые метрики и графики запусков.
- `shap_analysis.py`, `linear_regression_all_features.py`, `poly_regression_compare.py` — интерпретация и простые аппроксимации.

## Данные

Колонки:

- `Cod`, `Date`, `T`
- `Temperature_2m`, `Dewpoint_2m`, `Surface_pressure`, `Total_precipitation`, `Evaporation`
- `LST_Day`, `LST_Night`
- `X_final`, `Y_final`

Доли пропусков:

- `T`: `73.00%`
- `LST_Day`: `61.18%`
- `LST_Night`: `59.90%`
- ERA5-признаки: `12.33%`

## Хронология экспериментов (по фактическим run-папкам)

Ключевые тестовые метрики (`2022-2023`):

| Run | Эксперимент | R2 | RMSE | MAE |
|---|---|---:|---:|---:|
| `20250905_142927` | Базовый time-split (`xgb_optuna_timesplit.py`) | 0.9684 | 2.0564 | 0.9750 |
| `20250914_214644_extra` | Extra-features (`xgb_optuna_with_extra_features.py`) | 0.9868 | 1.3300 | 0.8492 |
| `20250915_165542_extra_v2` | Extra v2 (`xgb_optuna_with_extra_features_v2.py`) | 0.9868 | 1.3300 | 0.8482 |
| `20250916_154740_lags` | Лаг `t-1` (`xgb_optuna_with_lags.py`) | 0.9875 | 1.2945 | 0.7963 |
| `20250916_160311_lags123` | Лаги `t-1,t-2,t-3` (`xgb_optuna_with_lags123.py`) | 0.9890 | 1.2122 | 0.7634 |
| `20250916_171729_lags123_spatial` | + spatial (`xgb_optuna_with_lags123_spatial.py`) | 0.9898 | 1.1675 | 0.7189 |
| `20250916_173641_lags123_spatial_bias` | + bias correction (`xgb_optuna_with_lags123_spatial_bias.py`) | 0.9897 | 1.1773 | 0.7239 |
| `20250916_184426_lags123_spatial_longrun_ens5` | longrun ensemble (`xgb_optuna_with_lags123_spatial_longrun_ens5.py`) | 0.9894 | 1.1927 | 0.7286 |
| `20250923_114911_error_map` | spatial + error-map diagnostics | 0.9890 | 1.2124 | 0.7366 |
| `20250923_172831_resid_acf_pacf` | residual ACF/PACF + Ljung-Box | 0.9892 | 1.2037 | 0.7343 |
| `20250923_111926_winter_only` | winter-only режим | 0.9514 | 1.4540 | 0.8516 |
| `20250923_192103_without35108` | эксперимент без станции `35108` | 0.9954 | 0.7774 | 0.5920 |

Дополнительно есть инференс на полном массиве:

- `20250906_104013_infer` (`xgb_infer_full.py`): `R2=0.9391`, `RMSE=3.0115`
- `20250914_220035_infer_extra_fix` (`xgb_infer_full_extra.py`): `R2=0.9875`, `RMSE=1.3664`
- `20250915_172905_infer_extra_v2` (`xgb_infer_full_extra_v2.py`): `R2=0.9876`, `RMSE=1.3597`

## Структура скриптов

- EDA: `eda.py`
- Базовая модель: `xgb_optuna_timesplit.py`, `xgb_infer_full.py`
- Extra features: `xgb_optuna_with_extra_features.py`, `xgb_optuna_with_extra_features_v2.py`, `xgb_infer_full_extra.py`, `xgb_infer_full_extra_v2.py`, `xgb_optuna_with_extra_features_seasonal.py`
- Lags/Spatial: `xgb_optuna_with_lags.py`, `xgb_optuna_with_lags123.py`, `xgb_optuna_with_lags123_spatial.py`, `xgb_optuna_with_lags123_spatial_bias.py`, `xgb_optuna_with_lags123_spatial_longrun.py`, `xgb_optuna_with_lags123_spatial_longrun_ens5.py`, `xgb_optuna_with_lags123_spatial_exclude35108.py`
- Диагностика: `xgb_optuna_with_error_map.py`, `xgb_optuna_with_resid_acf_pacf.py`, `postcal_station_bias.py`
- Интерпретация/аппроксимации: `shap_analysis.py`, `linear_regression_all_features.py`, `poly_regression_compare.py`

## Как запустить

1. Установить зависимости:

```bash
pip install pandas numpy xgboost optuna scikit-learn matplotlib seaborn tqdm joblib shap statsmodels
```

2. Базовый сценарий:

```bash
python eda.py
python xgb_optuna_timesplit.py
python xgb_infer_full.py
python shap_analysis.py
```

3. Основная продвинутая ветка:

```bash
python xgb_optuna_with_lags123_spatial.py
python xgb_optuna_with_resid_acf_pacf.py
```

## Важные примечания

- В репозитории намеренно не хранятся тяжёлые сырые артефакты (`predictions.csv`, `residuals_all.csv`, модели), чтобы не раздувать историю git.
- Многие скрипты настроены на GPU (`device=cuda`); для CPU потребуется вручную поменять параметры XGBoost.
- Результаты из `outputs_runs/` следует воспринимать как журнал экспериментов; для новых запусков создаются новые timestamp-папки.
