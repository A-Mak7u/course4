import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import matplotlib.pyplot as plt

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

top_features = [
    "Temperature_2m", "Evaporation", "LST_Day", "Dewpoint_2m",
    "LST_Night", "Surface_pressure", "X_final", "Y_final", "Total_precipitation"
]
X_top = X[top_features]

degrees = [1, 2, 3]
results = []

for degree in tqdm(degrees, desc="Обработка полиномов"):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_top)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    n_terms = X_poly.shape[1]

    results.append((degree, r2, n_terms))

    print(f"\n📊 Полиномиальная регрессия степени {degree}")
    print(f"R² = {r2:.4f}, число членов = {n_terms}")
    feature_names = poly.get_feature_names_out(top_features)
    for f, c in zip(feature_names[:15], model.coef_[:15]):
        print(f"{f}: {c:.4f}")
    if len(feature_names) > 15:
        print("... (обрезано, всего коэффициентов:", len(feature_names), ")")
    print("Свободный член:", model.intercept_)

fig, ax1 = plt.subplots(figsize=(7, 5))

ax2 = ax1.twinx()
degrees_list = [d for d, _, _ in results]
r2_list = [r for _, r, _ in results]
terms_list = [n for _, _, n in results]

ax1.plot(degrees_list, r2_list, "o-r", label="R²")
ax2.plot(degrees_list, terms_list, "o-b", label="Число членов")

ax1.set_xlabel("Степень полинома")
ax1.set_ylabel("R²", color="r")
ax2.set_ylabel("Число членов", color="b")

plt.title("Сложность уравнения vs Точность")
plt.tight_layout()
plt.savefig("poly_regression_comparison.png")
plt.close()

print("\n✅ Результаты сохранены в poly_regression_comparison.png")
