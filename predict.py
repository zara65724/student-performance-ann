"""
predict.py — Command-line prediction tool
Usage: python predict.py
"""

import numpy as np
import joblib
from pathlib import Path

base   = Path(__file__).parent
model  = joblib.load(base / "model.joblib")
scaler = joblib.load(base / "scaler.joblib")


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict Pass/Fail for a new student.

    Parameters
    ----------
    attendance   : float  (0-100)
    assignment   : float  (0-100)
    quiz         : float  (0-100)
    mid          : float  (0-100)
    study_hours  : float  (hours per week)

    Returns
    -------
    dict: { result, label, probability }
    """
    features    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    features_sc = scaler.transform(features)
    prediction  = model.predict(features_sc)[0]
    probability = model.predict_proba(features_sc)[0][1]
    return {
        "result":      int(prediction),
        "label":       "Pass ✅" if prediction == 1 else "Fail ❌",
        "probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║  ANN Student Performance Evaluator  ║")
    print("╚══════════════════════════════════════╝\n")
    try:
        attendance  = float(input("Attendance   (0–100) : "))
        assignment  = float(input("Assignment   (0–100) : "))
        quiz        = float(input("Quiz         (0–100) : "))
        mid         = float(input("Mid-term     (0–100) : "))
        study_hours = float(input("Study hours / week   : "))
    except ValueError:
        print("❌ Please enter valid numbers only.")
        raise SystemExit(1)

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)
    print(f"\n{'─'*40}")
    print(f"  Predicted Result : {result['label']}")
    print(f"  Pass Probability : {result['probability'] * 100:.1f}%")
    print(f"{'─'*40}")
