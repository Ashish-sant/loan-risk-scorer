import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_prep import get_prepared_data

df = get_prepared_data()
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

joblib.dump(rf, "model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("saved model.pkl and feature_names.pkl")