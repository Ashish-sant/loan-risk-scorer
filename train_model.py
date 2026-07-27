import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from data_prep import get_prepared_data


def split_data(df):
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate(name, y_test, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print(classification_report(y_test, y_pred, digits=3))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    df = get_prepared_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # logistic regression (scaled)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)
    lr_pred = logreg.predict(X_test_s)
    lr_proba = logreg.predict_proba(X_test_s)[:, 1]
    evaluate("Logistic Regression", y_test, lr_pred, lr_proba)

    # random forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    evaluate("Random Forest", y_test, rf_pred, rf_proba)

    # top risk factors
    print("\n--- Top Risk Factors ---")
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    print(importance.sort_values(ascending=False).round(3).to_string())