import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1.2)

DATA_PATH = "final_2013_2023_T_ERA5_LST_daynight.csv"
OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Форма:", df.shape)
print("Колонки:", df.columns.tolist())

for col in ["Temperature_2m", "Dewpoint_2m"]:
    if col in df.columns:
        df[col] = df[col] - 273.15

na_share = df.isna().mean().sort_values(ascending=False)
print("\nПропуски (%):\n", na_share.head(12) * 100)

print("\nСтатистика:\n", df.describe().T)

num_cols = ["T", "Temperature_2m", "Dewpoint_2m",
            "Surface_pressure", "Total_precipitation",
            "Evaporation", "LST_Day", "LST_Night"]

for col in num_cols:
    if col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col].dropna(), bins=50, kde=True)
        plt.title(f"Распределение {col}")
        plt.xlabel(col)
        plt.ylabel("Частота")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/hist_{col}.png")
        plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[num_cols], orient="h")
plt.title("Boxplot ключевых переменных")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/boxplots.png")
plt.close()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    plt.figure(figsize=(10, 5))
    df.groupby("year")["T"].mean().plot(marker="o", label="Станции (T)")
    df.groupby("year")["Temperature_2m"].mean().plot(marker="o", label="ERA5")
    df.groupby("year")["LST_Day"].mean().plot(marker="o", label="MODIS LST Day")
    plt.title("Средняя температура по годам")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/temp_by_year.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    df.groupby("month")["T"].mean().plot(marker="o", label="Станции (T)")
    df.groupby("month")["Temperature_2m"].mean().plot(marker="o", label="ERA5")
    df.groupby("month")["LST_Day"].mean().plot(marker="o", label="MODIS LST Day")
    plt.title("Средняя температура по месяцам (сезонность)")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/temp_by_month.png")
    plt.close()

corr_cols = ["T", "Temperature_2m", "Dewpoint_2m", "Surface_pressure", "LST_Day", "LST_Night"]
corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Корреляции переменных")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_matrix.png")
plt.close()

print("\n✅ EDA завершен. Графики сохранены в папку", OUT_DIR)
