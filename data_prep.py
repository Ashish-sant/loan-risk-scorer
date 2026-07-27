import pandas as pd


def load_and_clean(path="data/cs-training.csv"):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # some ages are 0, replace with median
    median_age = df["age"].median()
    df.loc[df["age"] < 18, "age"] = median_age

    # ~20% of income is missing, fill with median (skewed data)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # cap extreme outliers at 99th percentile
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    return df


def engineer_features(df):
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    return df


def get_prepared_data(path="data/cs-training.csv"):
    df = load_and_clean(path)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = get_prepared_data()
    print("shape:", df.shape)
    print("missing:", df.isnull().sum().sum())
    print("default rate:", round(df["SeriousDlqin2yrs"].mean() * 100, 2), "%")