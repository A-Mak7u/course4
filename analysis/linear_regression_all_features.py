import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "final_2013_2023_T_ERA5_LST_daynight.csv"

df = pd.read_csv(DATA_PATH)

for col in ["Temperature_2m", "Dewpoint_2m"]:
    if col in df.columns:
        df[col] = df[col] - 273.15

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year

target = "T"

features = [c for c in df.columns if c not in ["Cod", "Date", "year", "T"]]
df = df.dropna(subset=[target])
X = df[features].fillna(-999)
y = df[target]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

print("\n📊 Линейная регрессия на всех признаках")
print("R²:", round(r2, 4))
print("Коэффициенты:")
for f, c in zip(features, model.coef_):
    print(f"{f}: {c:.4f}")
print("Свободный член:", model.intercept_)
