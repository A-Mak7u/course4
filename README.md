# Course4: журнал развития модели восстановления температуры `T`

Ниже не обзор проекта, а именно история итераций: что менялось в моделях, какие метрики получались и чем это подтверждалось на графиках.

---

## 1. Исходные данные и EDA

- Период: `2013-2023`
- Объём: `208278` строк
- Целевая: `T`
- Основные источники: ERA5 (`Temperature_2m`, `Dewpoint_2m`, `Surface_pressure`, `Evaporation`, `Total_precipitation`) + MODIS (`LST_Day`, `LST_Night`)

Пропуски:

- `T`: `73.00%`
- `LST_Day`: `61.18%`
- `LST_Night`: `59.90%`
- ERA5-признаки: `12.33%`

Графики EDA:

<p align="center">
  <img src="eda_plots/correlation_matrix.png" width="650">
</p>

<p align="center">
  <img src="eda_plots/temp_by_year.png" width="420">
  <img src="eda_plots/temp_by_month.png" width="420">
</p>

Ключевой вывод этапа: `Temperature_2m` максимально связан с `T`, но для устойчивого качества по сезонам нужны дополнительные нелинейные и временные признаки.

---

## 2. Базовый XGBoost (time split)

Скрипт: `xgb_optuna_timesplit.py`  
Run: `outputs_runs/20250905_142927`

Что было сделано:

- Train: `2013-2021`, test: `2022-2023`
- Optuna-тюнинг для базового набора признаков без лагов/пространства

Метрики test:

- `R2 = 0.9684`
- `RMSE = 2.0564`
- `MAE = 0.9750`
- `MedAE = 0.5961`

<p align="center">
  <img src="outputs_runs/20250905_142927/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250905_142927/residuals_hist.png" width="420">
</p>

Инференс этой же ветки на полном массиве (`xgb_infer_full.py`, run `20250906_104013_infer`):

- `R2 = 0.9391`
- `RMSE = 3.0115`
- `MAE = 1.2429`
- `SMAPE = 5.5845`

---

## 3. Интерпретация и аппроксимации

Скрипты: `shap_analysis.py`, `linear_regression_all_features.py`, `poly_regression_compare.py`

Что добавилось:

- SHAP-разбор важности признаков
- Проверка, насколько линейные/полиномиальные формулы могут приблизить поведение модели

<p align="center">
  <img src="shap_summary.png" width="430">
  <img src="shap_importance.png" width="430">
</p>

<p align="center">
  <img src="poly_regression_comparison.png" width="520">
</p>

Итог: линейная модель даёт адекватный baseline интерпретации, но для качества уровня production нужна нелинейная XGBoost-ветка с инженерией признаков.

---

## 4. Extra-features ветка

Скрипт: `xgb_optuna_with_extra_features.py`  
Run: `outputs_runs/20250914_214644_extra`

Изменения в признаках:

- сезонные: `dayofyear`, `sin_doy`, `cos_doy`
- физически-инженерные: `dewpoint_dep = Temperature_2m - Dewpoint_2m`
- суточная амплитуда: `diurnal_range = LST_Day - LST_Night`

Метрики test:

- `R2 = 0.9868`
- `RMSE = 1.3300`
- `MAE = 0.8492`
- `MedAE = 0.5919`

<p align="center">
  <img src="outputs_runs/20250914_214644_extra/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250914_214644_extra/residuals_hist.png" width="420">
</p>

Инференс на полном массиве (`xgb_infer_full_extra.py`, run `20250914_220035_infer_extra_fix`):

- `R2 = 0.9875`
- `RMSE = 1.3664`
- `MAE = 0.8534`
- `MedAE = 0.5735`

<p align="center">
  <img src="outputs_runs/20250914_220035_infer_extra_fix/density_true_vs_pred.png" width="520">
</p>

Ключевой эффект этапа: самый крупный скачок качества относительно baseline.

---

## 5. Сезонное разделение как отдельный эксперимент

Скрипт: `xgb_optuna_with_extra_features_seasonal.py`  
Run: `outputs_runs/20250915_094512_seasonal`

Что проверялось: раздельное обучение для холодного и тёплого сезона.

Результаты test:

- Cold (11-3): `R2 = 0.9450`, `RMSE = 1.5472`, `MAE = 0.9546`
- Warm (4-10): `R2 = 0.9708`, `RMSE = 1.1383`, `MAE = 0.7684`

Итог: разбиение по сезонам оказалось хуже единой extra-модели.

---

## 6. Посткоррекция смещения по станциям

Скрипт: `postcal_station_bias.py`  
Run: `outputs_runs/20250915_101542_biascorr`

Идея: после инференса вычитать station-wise bias, рассчитанный по train.

До:

- `R2 = 0.9875`
- `RMSE = 1.3664`
- `MAE = 0.8534`
- `MedAE = 0.5735`

После:

- `R2 = 0.9875`
- `RMSE = 1.3661`
- `MAE = 0.8529`
- `MedAE = 0.5722`

<p align="center">
  <img src="outputs_runs/20250915_101542_biascorr/scatter_biascorr.png" width="520">
</p>

Итог: улучшение минимальное, как универсальный шаг не закрепилось.

---

## 7. Extra v2

Скрипт: `xgb_optuna_with_extra_features_v2.py`  
Run: `outputs_runs/20250915_165542_extra_v2`

Что изменено:

- доработан состав признаков (включая календарные)
- стабилизирован инференс и отчётность по группам

Метрики test:

- `R2 = 0.9868`
- `RMSE = 1.3300`
- `MAE = 0.8482`
- `MedAE = 0.5871`

Полный инференс (`xgb_infer_full_extra_v2.py`, run `20250915_172905_infer_extra_v2`):

- `R2 = 0.9876`
- `RMSE = 1.3597`
- `MAE = 0.8448`
- `MedAE = 0.5649`

<p align="center">
  <img src="outputs_runs/20250915_172905_infer_extra_v2/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250915_172905_infer_extra_v2/residuals_hist.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250915_172905_infer_extra_v2/boxplot_error_by_month.png" width="520">
</p>

---

## 8. Переход к лагам

### 8.1 Лаг `t-1`

Скрипт: `xgb_optuna_with_lags.py`  
Run: `outputs_runs/20250916_154740_lags`

Test:

- `R2 = 0.9875`
- `RMSE = 1.2945`
- `MAE = 0.7963`

### 8.2 Лаги `t-1, t-2, t-3`

Скрипт: `xgb_optuna_with_lags123.py`  
Run: `outputs_runs/20250916_160311_lags123`

Test:

- `R2 = 0.9890`
- `RMSE = 1.2122`
- `MAE = 0.7634`

Фиксированный прогон этой же ветки: `outputs_runs/20250916_163343_lags123_fix`

- `R2 = 0.9896`
- `RMSE = 1.1816`
- `MAE = 0.7452`

<p align="center">
  <img src="outputs_runs/20250916_163343_lags123_fix/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250916_163343_lags123_fix/residuals_hist.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250916_163343_lags123_fix/density_true_vs_pred_full.png" width="520">
</p>

Итог этапа: лаги дали устойчивый прирост, особенно по зимним режимам.

---

## 9. Лаги + spatial-фичи

Скрипт: `xgb_optuna_with_lags123_spatial.py`  
Run: `outputs_runs/20250916_171729_lags123_spatial`

Что добавлено поверх лагов:

- `sin/cos(lat)`, `sin/cos(lon)`
- `station_train_mean_T`

Test:

- `R2 = 0.9898`
- `RMSE = 1.1675`
- `MAE = 0.7189`
- `MedAE = 0.4992`

<p align="center">
  <img src="outputs_runs/20250916_171729_lags123_spatial/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250916_171729_lags123_spatial/density_true_vs_pred_full.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250916_171729_lags123_spatial/boxplot_error_by_month.png" width="520">
</p>

---

## 10. Проверка post-bias поверх spatial

Скрипт: `xgb_optuna_with_lags123_spatial_bias.py`  
Run: `outputs_runs/20250916_173641_lags123_spatial_bias`

До коррекции:

- `R2 = 0.989652`
- `RMSE = 1.177280`
- `MAE = 0.723921`

После коррекции:

- `R2 = 0.989652`
- `RMSE = 1.177295`
- `MAE = 0.723917`

<p align="center">
  <img src="outputs_runs/20250916_173641_lags123_spatial_bias/scatter_pred_vs_true_before.png" width="420">
  <img src="outputs_runs/20250916_173641_lags123_spatial_bias/scatter_pred_vs_true.png" width="420">
</p>

Итог: практически нулевой эффект, шаг оставлен как диагностический, не как основной.

---

## 11. Long-run и ансамбль из 5 моделей

### 11.1 Long-run бустинг

Скрипт: `xgb_optuna_with_lags123_spatial_longrun.py`  
Run: `outputs_runs/20250916_180430_lags123_spatial_longrun`

Test:

- `R2 = 0.9893`
- `RMSE = 1.1961`
- `MAE = 0.7304`

### 11.2 Ensemble (5 seeds)

Скрипт: `xgb_optuna_with_lags123_spatial_longrun_ens5.py`  
Run: `outputs_runs/20250916_184426_lags123_spatial_longrun_ens5`

Test:

- `R2 = 0.9894`
- `RMSE = 1.1927`
- `MAE = 0.7286`

<p align="center">
  <img src="outputs_runs/20250916_184426_lags123_spatial_longrun_ens5/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250916_184426_lags123_spatial_longrun_ens5/residuals_hist.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250916_184426_lags123_spatial_longrun_ens5/boxplot_error_by_month.png" width="520">
</p>

Итог: стабильность выросла, но прорыва относительно `lags123_spatial` не произошло.

---

## 12. Winter-only режим

Скрипт: `xgb_optuna_winter_only.py`  
Run: `outputs_runs/20250923_111926_winter_only`

Test (только зимние месяцы):

- `R2 = 0.9514`
- `RMSE = 1.4540`
- `MAE = 0.8516`

<p align="center">
  <img src="outputs_runs/20250923_111926_winter_only/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250923_111926_winter_only/residuals_hist.png" width="420">
</p>

Итог: подтверждена сложность зимнего режима как отдельной задачи.

---

## 13. Диагностика station-wise ошибок (карты)

Скрипт: `xgb_optuna_with_error_map.py`  
Run: `outputs_runs/20250923_114911_error_map`

Test:

- `R2 = 0.9890`
- `RMSE = 1.2124`
- `MAE = 0.7366`

<p align="center">
  <img src="outputs_runs/20250923_114911_error_map/map_bias_test.png" width="420">
  <img src="outputs_runs/20250923_114911_error_map/map_mae_test.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250923_114911_error_map/map_bias_full.png" width="420">
  <img src="outputs_runs/20250923_114911_error_map/map_mae_full.png" width="420">
</p>

Топ-5 станций по `MAE` на test:

- Лучшие: `34059`, `34240`, `34152`, `34391`, `27962`
- Худшие: `35108`, `35007`, `27857`, `27995`, `34289`

Вывод: станция `35108` явно выбивается как проблемная.

---

## 14. Диагностика автокорреляции остатков

Скрипт: `xgb_optuna_with_resid_acf_pacf.py`  
Run: `outputs_runs/20250923_172831_resid_acf_pacf`

Test:

- `R2 = 0.9892`
- `RMSE = 1.2037`
- `MAE = 0.7343`

<p align="center">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/scatter_pred_vs_true.png" width="420">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/residuals_hist.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/resid_test_acf.png" width="420">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/resid_test_pacf.png" width="420">
</p>

<p align="center">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/resid_test_winter_acf.png" width="420">
  <img src="outputs_runs/20250923_172831_resid_acf_pacf/resid_test_winter_pacf.png" width="420">
</p>

Смысл этапа: проверить, сколько временной структуры остаётся в residuals после пространственно-лаговой модели.

---

## 15. Эксперимент по станции 35108 (включать/исключать)

Скрипт: `xgb_optuna_with_lags123_spatial_exclude35108.py`

С 35108 (`outputs_runs/20250923_191322_with35108`):

- `R2 = 0.9894`
- `RMSE = 1.1913`
- `MAE = 0.7281`

Без 35108 (`outputs_runs/20250923_192103_without35108`):

- `R2 = 0.9954`
- `RMSE = 0.7774`
- `MAE = 0.5920`

Итог: сильнейший прирост метрик при исключении 35108, то есть это отдельный сложный домен/станция, а не случайный шум.

---

## 16. Сводка прогресса (test 2022-2023)

| Этап | R2 | RMSE | MAE |
|---|---:|---:|---:|
| Baseline (`20250905_142927`) | 0.9684 | 2.0564 | 0.9750 |
| Extra-features (`20250914_214644_extra`) | 0.9868 | 1.3300 | 0.8492 |
| Lags t-1 (`20250916_154740_lags`) | 0.9875 | 1.2945 | 0.7963 |
| Lags t-1..t-3 (`20250916_163343_lags123_fix`) | 0.9896 | 1.1816 | 0.7452 |
| Lags + spatial (`20250916_171729_lags123_spatial`) | 0.9898 | 1.1675 | 0.7189 |
| Longrun ens5 (`20250916_184426_lags123_spatial_longrun_ens5`) | 0.9894 | 1.1927 | 0.7286 |
| Без станции 35108 (`20250923_192103_without35108`) | 0.9954 | 0.7774 | 0.5920 |

Главная траектория развития:  
`baseline -> extra-features -> lag features -> spatial features -> диагностика проблемных станций`.
