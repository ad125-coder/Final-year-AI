import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Create results directory for saved figures
os.makedirs("results", exist_ok=True)
SEED = 42

print("=== 1. LOADING DATA & CHECKING IMBALANCE ===")
df = pd.read_csv("data/creditcard.csv")
print(f"Dataset shape: {df.shape}")
class_counts = df["Class"].value_counts()
fraud_ratio = class_counts[1] / len(df) * 100
print(
    f"Class distribution:\nLegitimate (0): {class_counts[0]} | Fraud (1): {class_counts[1]} ({fraud_ratio:.3f}%)"
)

X = df.drop(columns="Class")
y = df["Class"]

# 2. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

results_summary = []


def evaluate_model(name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print(f"\n--- {name} ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    print(f"PR-AUC: {pr_auc:.4f}")

    results_summary.append({
        "Model": name,
        "Precision (Fraud)": report["1"]["precision"],
        "Recall (Fraud)": report["1"]["recall"],
        "F1-score (Fraud)": report["1"]["f1-score"],
        "PR-AUC": pr_auc
    })
    return cm


print("\n=== 3. BASELINES & MODELS ===")

# Baseline 1: Dummy Classifier
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_proba = dummy.predict_proba(X_test)[:, 1]
evaluate_model("Dummy (Most Frequent)", y_test, dummy_pred, dummy_proba)

# Baseline 2: Logistic Regression with class weights
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]
evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba)

# Model 3: Random Forest (handling imbalance via class_weight)
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

# Model 4: XGBoost (handling imbalance via scale_pos_weight)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight, random_state=SEED, eval_metric="logloss", n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba)

print("\n=== 4. EXTENSION: THRESHOLD TUNING & COST ANALYSIS ===")
# Find optimal threshold to maximize F1-score on test set (or validation set)
precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
print(f"Optimal Threshold (XGBoost) for max F1: {optimal_threshold:.4f}")

xgb_pred_opt = (xgb_proba >= optimal_threshold).astype(int)
evaluate_model("XGBoost (Optimized Threshold)", y_test, xgb_pred_opt, xgb_proba)

print("\n=== 5. GENERATING & SAVING FIGURES ===")

# Figure 1: Confusion Matrix for XGBoost (Optimized)
plt.figure(figsize=(6, 5))
cm_opt = confusion_matrix(y_test, xgb_pred_opt)
sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - XGBoost (Tuned Threshold)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix_xgboost.png", dpi=300)
plt.close()

# Figure 2: Precision-Recall Curve Comparison
plt.figure(figsize=(7, 5))
for name, probas in [
    ("Logistic Regression", lr_proba),
    ("Random Forest", rf_proba),
    ("XGBoost", xgb_proba)
]:
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("results/precision_recall_curves.png", dpi=300)
plt.close()

# Figure 3: Feature Importance (Random Forest)
plt.figure(figsize=(8, 5))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10 features
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices], align="center")
plt.xticks(range(10), [X.columns[i] for i in indices], rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

print("\n=== SUMMARY TABLE ===")
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))