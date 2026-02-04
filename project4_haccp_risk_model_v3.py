import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv("data/haccp_compliance_regression_wafercream_2023_2024.csv")

target_col = "Noncompliance"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

y = df[target_col]
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_auc_scores = []
rf_auc_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

    pipe_log = Pipeline([
        ("prep", preprocessor),
        ("model", log_reg)
    ])
    pipe_log.fit(X_train_cv, y_train_cv)
    y_prob_log = pipe_log.predict_proba(X_test_cv)[:, 1]
    log_auc_scores.append(roc_auc_score(y_test_cv, y_prob_log))

    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("model", rf)
    ])
    pipe_rf.fit(X_train_cv, y_train_cv)
    y_prob_rf = pipe_rf.predict_proba(X_test_cv)[:, 1]
    rf_auc_scores.append(roc_auc_score(y_test_cv, y_prob_rf))

print("===== CROSS-VALIDATION RESULTS (5-fold) =====")
print(f"Logistic Regression AUC: {np.mean(log_auc_scores):.3f} ± {np.std(log_auc_scores):.3f}")
print(f"Random Forest AUC:       {np.mean(rf_auc_scores):.3f} ± {np.std(rf_auc_scores):.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

pipe_log_smote = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", log_reg)
])

pipe_log_smote.fit(X_train, y_train)
y_prob = pipe_log_smote.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_j = tpr - fpr
best_threshold = thresholds[np.argmax(youden_j)]

print("\n===== THRESHOLD TUNING =====")
print(f"Best threshold (Youden J): {best_threshold:.3f}")

y_pred_best = (y_prob >= best_threshold).astype(int)

print("\n===== CLASSIFICATION REPORT (Optimized threshold) =====")
print(classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)

auc_final = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"Logistic+SMOTE (AUC={auc_final:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — HACCP Noncompliance Risk Model — Niusha Portfolio (2024)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_logistic_cv.png", dpi=300)
plt.close()

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Optimized threshold)")
plt.colorbar()
plt.xticks([0, 1], ["Compliant", "Noncompliant"])
plt.yticks([0, 1], ["Compliant", "Noncompliant"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

df_test = X_test.copy()
df_test["predicted_risk"] = y_prob
df_test["true_label"] = y_test.values

if "Area" in df_test.columns:
    risk_by_area = df_test.groupby("Area")["predicted_risk"].mean().sort_values()
    plt.figure(figsize=(10, 6))
    risk_by_area.plot(kind="barh")
    plt.xlabel("Avg Predicted Risk")
    plt.title("Average Predicted Noncompliance Risk by Area")
    plt.tight_layout()
    plt.savefig("outputs/risk_by_area.png", dpi=300)
    plt.close()

print("\n✅ Outputs saved in /outputs folder.")

print("Finished ✅ — results saved in outputs/")
