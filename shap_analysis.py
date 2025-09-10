import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
import numpy as np

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
MODEL_PATH = "outputs_runs/20250905_142927/xgb_model.pkl"

df = pd.read_csv(DATA_PATH)

for col in ["Temperature_2m", "Dewpoint_2m"]:
    if col in df.columns:
        df[col] = df[col] - 273.15

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year

features = [c for c in df.columns if c not in ["Cod", "Date", "year", "T"]]
df = df.dropna(subset=["T"])
X = df[features].fillna(-999)
y = df["T"]

model = joblib.load(MODEL_PATH)
model.set_param({"device": "cpu"})

X_sample = X.sample(3000, random_state=42)

explainer = shap.TreeExplainer(model)

print("⚡ Считаем SHAP-значения для 3000 точек...")
shap_values_list = []
batch_size = 500  # считаем партиями, чтобы было видно прогресс

for i in tqdm(range(0, X_sample.shape[0], batch_size)):
    batch = X_sample.iloc[i:i+batch_size]
    shap_values_batch = explainer.shap_values(batch)
    shap_values_list.append(shap_values_batch)

shap_values = np.vstack(shap_values_list)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance.png")
plt.close()

print("✅ SHAP графики сохранены: shap_summary.png, shap_importance.png")

importances = pd.DataFrame({
    "feature": X_sample.columns,
    "importance": abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

top5 = importances.head(5)["feature"].tolist()
print("Топ-5 признаков:", top5)

X_top5 = X[top5]
linreg = LinearRegression()
linreg.fit(X_top5, y)
y_pred = linreg.predict(X_top5)

print("\n📊 Линейная аппроксимация:")
print("R2:", r2_score(y, y_pred))
print("Коэффициенты:")
for f, c in zip(top5, linreg.coef_):
    print(f"{f}: {c:.4f}")
print("Свободный член:", linreg.intercept_)


'''
Топ-5 признаков: ['Temperature_2m', 'Evaporation', 'LST_Day', 'Dewpoint_2m', 'LST_Night']

📊 Линейная аппроксимация:
R2: 0.8325887064848116
Коэффициенты:
Temperature_2m: 0.9231
Evaporation: -1.0194
LST_Day: 0.0013
Dewpoint_2m: 0.0893
LST_Night: 0.0000
Свободный член: 1.2352101898931904
'''