"""
train_ann.py
Run this locally to retrain the model and regenerate model.joblib + scaler.joblib.
Usage: python train_ann.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

print("=" * 60)
print("  ANN Student Performance Trainer")
print("=" * 60)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_excel("dataset.xlsx")
print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}\n")
print(f"Class distribution:\n{df['result'].value_counts()}\n")

FEATURES = ["attendance", "assignment", "quiz", "mid", "study_hours"]
TARGET   = "result"

X = df[FEATURES].values
y = df[TARGET].values

# ── Split & Scale ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"Train size : {X_train_sc.shape[0]}")
print(f"Test size  : {X_test_sc.shape[0]}\n")

# ── Build & Train ANN ──────────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False,
)

print("Training ANN...")
model.fit(X_train_sc, y_train)
print(f"Converged after {model.n_iter_} iterations.")
print(f"Best val score : {model.best_validation_score_:.4f}\n")

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)
print(f"Test Accuracy : {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# ── Plot & Save ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0f172a")

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(ax=axes[0], colorbar=False, cmap="Purples")
axes[0].set_title("Confusion Matrix", color="white")
axes[0].set_facecolor("#1e293b")
axes[0].tick_params(colors="white")
axes[0].xaxis.label.set_color("white")
axes[0].yaxis.label.set_color("white")

axes[1].plot(model.loss_curve_, color="#6366f1", linewidth=2, label="Training Loss")
axes[1].set_facecolor("#1e293b")
axes[1].set_xlabel("Iteration", color="white")
axes[1].set_ylabel("Loss", color="white")
axes[1].set_title("Training Loss Curve", color="white")
axes[1].tick_params(colors="white")
axes[1].legend()
for spine in axes[1].spines.values():
    spine.set_color("#334155")

plt.tight_layout()
plt.savefig("training_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → training_report.png")

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved → model.joblib")
print("Saved → scaler.joblib")
print("\n✅ Training complete! Run: streamlit run app.py")


# ── Reusable evaluation function ──────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """Predict Pass/Fail for a student using the trained ANN."""
    features    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    features_sc = scaler.transform(features)
    prediction  = model.predict(features_sc)[0]
    probability = model.predict_proba(features_sc)[0][1]
    return {
        "result":      int(prediction),
        "label":       "Pass ✅" if prediction == 1 else "Fail ❌",
        "probability": round(float(probability), 4),
    }
