import shap
import matplotlib.pyplot as plt
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

# SHAP on a sample (full 30k is slow; 500 is enough to see patterns)
sample = X_test.sample(500, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(sample)

# handle sklearn's output shape (list for the two classes)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

# overall summary: which features drive risk, and in which direction
shap.summary_plot(sv, sample, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved outputs/shap_summary.png")

# explain ONE applicant
i = 0
print("\nExplaining applicant:")
print(sample.iloc[i])