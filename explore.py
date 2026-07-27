import pandas as pd

df = pd.read_csv("data/cs-training.csv")

print("Shape (rows, columns):", df.shape)
print()

# How many missing values in each column, and what percent
print(" Missing values ")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
for col in df.columns:
    if missing[col] > 0:
        print(f"{col}: {missing[col]} missing ({missing_pct[col]}%)")
print()

# The target: how balanced is it?
print(" Target balance ")
print(df["SeriousDlqin2yrs"].value_counts())
print("Default rate:", round(df["SeriousDlqin2yrs"].mean() * 100, 2), "%")