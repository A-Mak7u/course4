## 1. Целевая таблица

- Регион: `Волгоградская область`
- Период: `2013-2023`
- Источник целевой: `Meteostat`
- Подтверждённые станции: `13`
- Подтверждённый target CSV: `data/volgograd/processed/volgograd_target_daily_meteostat_2013_2023.csv`
- Объём target: `34700` строк

Подтверждённые артефакты:

- `data/volgograd/processed/volgograd_station_metadata_meteostat.csv`
- `data/volgograd/processed/volgograd_target_daily_meteostat_2013_2023.csv`
- `data/volgograd/processed/volgograd_target_coverage_meteostat_2013_2023.csv`

Итог этапа: целевая таблица под Волгоград собрана в том же ключе `Cod / Date / X / Y / T`, что и исходная саратовская ветка.

---

## 2. Подтверждённый пайплайн признаков

Собранный код:

- `transfer/build_volgograd_targets_meteostat.py`
- `transfer/build_volgograd_era5_daily.py`
- `transfer/submit_volgograd_modis_task.py`
- `transfer/fetch_volgograd_modis_task.py`
- `transfer/parse_volgograd_modis_appeears.py`
- `transfer/build_volgograd_final_dataset.py`

Что уже подтверждено по логике сборки:

- `ERA5`: региональные hourly bbox-файлы -> интерполяция в точки станций -> суточное среднее
- `MODIS`: `AppEEARS point extraction` по `MOD11A1.061`, слои `LST_Day_1km` и `LST_Night_1km`
- итоговая склейка запланирована в формат саратовского датасета: `Cod, Date, T, ERA5, LST_Day, LST_Night, X_final, Y_final`

Итог этапа: под Волгоград уже есть воспроизводимый data-pipeline, совместимый с существующим `XGB transfer`-кодом.

---

## 3. Что уже подтверждено артефактами

### 3.1 ERA5 smoke test

Подтверждённый monthly-run:

- `data/volgograd/interim/era5_daily_yearly/volgograd_era5_daily_2014_01.csv`
- `403` строк = `13` станций x `31` день

Это подтверждает:

- корректную загрузку `ERA5`
- корректную распаковку monthly-архивов
- корректную интерполяцию в station points
- корректную суточную агрегацию

### 3.2 MODIS probe

Подтверждённый probe-task:

- `AppEEARS point results.csv` для станции `34560` за `2023`
- parser успешно преобразует результат в daily-table `Cod / Date / LST_Day / LST_Night`

Подтверждённые probe-артефакты:

- `tmp_modis_probe_fetch/volgograd-modis-probe-34560-2023-MOD11A1-061-results.csv`
- `tmp_modis_probe_parse/modis_probe_daily.csv`

Для probe-парсинга наблюдается типичное sparse-покрытие:

- `LST_Day` missing share ≈ `0.619`
- `LST_Night` missing share ≈ `0.622`

Итог этапа: формат `AppEEARS`-выгрузки подтверждён не по документации, а на реальном `results.csv`; parser уже работает на живом файле.

---

## 4. Текущее состояние полного прогона

Подтверждённые long-run процессы:

- полный `ERA5 2013-2023`
- большой `MODIS AppEEARS task` на `2013-2023`
- fallback-режим: `11` yearly `MODIS AppEEARS` tasks (`2013..2023`)
- watcher-suite под автозапуск transfer-тестов

Ключевые логи:

- `data/volgograd/processed/era5_full_build.log`
- `data/volgograd/processed/modis_full_fetch.log`
- `data/volgograd/processed/modis_yearly_fetch.log`
- `data/volgograd/processed/volgograd_suite.log`

Подтверждённый статус:

- `ERA5` накопливается помесячно и уже прошёл существенную часть периода
- один большой `MODIS`-task остаётся в `processing`
- yearly `MODIS` fallback уже отправлен по всем годам

Итог этапа: bottleneck проекта сейчас не в коде модели, а в завершении полного набора удалённых `MODIS`-выгрузок.

---

## 5. Модельная ветка

Подготовленный автоматический прогон:

- full target-train
- `fewshot_5`
- `fewshot_3`
- device: `CUDA`

Оркестратор:

- `transfer/wait_and_run_volgograd_suite.py`

Выходной run-bundle:

- `outputs_runs/20260327_185109_volgograd_transfer_suite`

Подтверждённый текущий статус:

- `xgboost` на `CUDA` в новой среде работает
- финальные transfer-метрики по Волгограду ещё не зафиксированы, потому что полный `Volgograd final CSV` пока не собран

Итог этапа: эксперименты под Волгоград не заблокированы кодом или окружением; они ожидают только полного прихода данных.
