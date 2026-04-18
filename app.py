"""
app.py — ANN Student Performance Evaluator
Streamlit Cloud deployment entry point.
"""

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Evaluator | ANN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.result-pass {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 8px 32px rgba(16,185,129,0.3);
    margin: 1rem 0;
}

.result-fail {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 8px 32px rgba(239,68,68,0.3);
    margin: 1rem 0;
}

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.4rem 0;
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}

.metric-lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

.info-box {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
}

.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ANN model…")
def load_artifacts():
    base = Path(__file__).parent
    model  = joblib.load(base / "model.joblib")
    scaler = joblib.load(base / "scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()


# ── Evaluation Function (Task 7) ───────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    features_sc = scaler.transform(features)
    prediction  = model.predict(features_sc)[0]
    probability = model.predict_proba(features_sc)[0][1]
    return {
        "result":      int(prediction),
        "label":       "PASS ✅" if prediction == 1 else "FAIL ❌",
        "probability": round(float(probability), 4),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About the Model")
    st.markdown("""
    <div class='info-box'>
    <b>Architecture:</b><br>
    Input(5) → Dense(64, ReLU)<br>
    → Dense(32, ReLU) → Output<br><br>
    <b>Optimizer:</b> Adam<br>
    <b>Accuracy:</b> 82.5%<br>
    <b>Dataset:</b> 600 students
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Feature Guide")
    guide = {
        "Attendance": "% of classes attended",
        "Assignment": "Assignment marks (0–100)",
        "Quiz": "Quiz marks (0–100)",
        "Mid-term": "Mid exam marks (0–100)",
        "Study Hours": "Hours studied per week",
    }
    for k, v in guide.items():
        st.markdown(f"**{k}** — {v}")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com)")
    st.markdown("[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)")


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Student Performance Evaluator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Artificial Neural Network — Binary Classification (Pass / Fail)</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Report", "📘 How It Works"])

# ─────────────────────────────────────────────────
# TAB 1: PREDICT
# ─────────────────────────────────────────────────
with tab1:
    st.markdown("#### Enter Student Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
        assignment = st.slider("📝 Assignment Score", 0, 100, 70)

    with col2:
        quiz = st.slider("❓ Quiz Score", 0, 100, 65)
        mid  = st.slider("📋 Mid-term Score", 0, 100, 60)

    with col3:
        study_hours = st.slider("⏰ Study Hours / Week", 0, 30, 6)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍  Predict Result", use_container_width=True)

    st.divider()

    if predict_btn:
        result = evaluate_student(attendance, assignment, quiz, mid, study_hours)
        prob   = result["probability"]

        # ── Result Banner ──────────────────────────────────────────────────────
        css_class = "result-pass" if result["result"] == 1 else "result-fail"
        st.markdown(
            f'<div class="{css_class}">Predicted Result: {result["label"]}</div>',
            unsafe_allow_html=True
        )

        # ── Metrics Row ────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{prob*100:.1f}%</div><div class="metric-lbl">Pass Probability</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{attendance}%</div><div class="metric-lbl">Attendance</div></div>', unsafe_allow_html=True)
        with m3:
            avg_score = round((assignment + quiz + mid) / 3, 1)
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_score}</div><div class="metric-lbl">Avg Score</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{study_hours}h</div><div class="metric-lbl">Study/Week</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Feature Profile**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            labels  = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study×5"]
            values  = [attendance, assignment, quiz, mid, min(study_hours * 5, 100)]
            colors  = ["#10b981" if v >= 50 else "#ef4444" for v in values]
            bars    = ax.barh(labels, values, color=colors, edgecolor="#1e293b", height=0.55)
            ax.set_xlim(0, 110)
            ax.axvline(50, color="#475569", linestyle="--", linewidth=0.8)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("Score / Scaled Value", color="#94a3b8", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("**Pass Probability Gauge**")
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            fig2.patch.set_facecolor("#0f172a")
            ax2.set_facecolor("#0f172a")
            color = "#10b981" if result["result"] == 1 else "#ef4444"
            ax2.barh(["Probability"], [prob], color=color, height=0.4)
            ax2.barh(["Probability"], [1 - prob], left=[prob], color="#1e293b", height=0.4)
            ax2.set_xlim(0, 1)
            ax2.axvline(0.5, color="#475569", linestyle="--", linewidth=0.8)
            ax2.text(prob / 2, 0, f"{prob*100:.1f}%", ha="center", va="center",
                     color="white", fontsize=16, fontweight="bold",
                     fontfamily="monospace")
            ax2.tick_params(colors="#94a3b8", labelsize=9)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Recommendation ─────────────────────────────────────────────────────
        st.markdown("**📌 Recommendation**")
        tips = []
        if attendance < 75:   tips.append("📌 Increase attendance above 75%")
        if assignment < 60:   tips.append("📌 Improve assignment submissions")
        if quiz < 50:         tips.append("📌 Practice more quizzes")
        if mid < 50:          tips.append("📌 Focus on mid-term preparation")
        if study_hours < 5:   tips.append("📌 Study at least 5 hours per week")

        if result["result"] == 1 and not tips:
            st.success("🌟 Excellent performance across all areas! Keep it up.")
        elif tips:
            for t in tips:
                st.warning(t)
        else:
            st.info("Performance is on track. Maintain consistency.")


# ─────────────────────────────────────────────────
# TAB 2: MODEL REPORT
# ─────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📊 Training Report")

    report_path = Path(__file__).parent / "training_report.png"
    if report_path.exists():
        st.image(str(report_path), caption="Confusion Matrix & Training Loss Curve", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [("82.5%", "Accuracy"), ("83%", "Precision"), ("82%", "Recall"), ("82%", "F1-Score")]
    for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📂 Dataset Overview")
    try:
        df = pd.read_excel(Path(__file__).parent / "dataset.xlsx")
        st.dataframe(df.describe().round(2), use_container_width=True)
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**Total students:** {len(df)}")
            st.markdown(f"**Features:** {list(df.columns[:-1])}")
        with colB:
            pass_count = (df['result'] == 1).sum()
            fail_count = (df['result'] == 0).sum()
            st.markdown(f"**Pass:** {pass_count} ({pass_count/len(df)*100:.1f}%)")
            st.markdown(f"**Fail:** {fail_count} ({fail_count/len(df)*100:.1f}%)")
    except Exception:
        st.info("Dataset file not found in deployment.")


# ─────────────────────────────────────────────────
# TAB 3: HOW IT WORKS
# ─────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🧠 What is an ANN?")
    st.markdown("""
    <div class='info-box'>
    An <b>Artificial Neural Network (ANN)</b> is a machine learning model inspired by the human brain.
    It learns patterns from data by adjusting internal weights through a process called <b>backpropagation</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔁 How This Model Works")
    steps = [
        ("1️⃣ Input", "5 student features are fed in: attendance, assignment, quiz, mid, study_hours"),
        ("2️⃣ Scaling", "StandardScaler normalises values to mean=0, std=1 so all features contribute equally"),
        ("3️⃣ Hidden Layer 1", "64 neurons with ReLU activation extract low-level patterns"),
        ("4️⃣ Hidden Layer 2", "32 neurons refine patterns into higher-level representations"),
        ("5️⃣ Output", "A sigmoid neuron outputs probability of Passing (0.0 → 1.0)"),
        ("6️⃣ Decision", "If probability ≥ 0.5 → PASS, else → FAIL"),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.markdown("#### ⚠️ Limitations")
    limitations = [
        "Trained on synthetic data — real students may show different patterns",
        "Only 5 features — motivation, background, etc. are not captured",
        "Binary output only — no grade prediction",
        "Black-box model — hard to explain individual predictions",
    ]
    for l in limitations:
        st.markdown(f"- {l}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
Built with ❤️ using <b>scikit-learn</b> + <b>Streamlit</b> &nbsp;|&nbsp;
MLPClassifier (64→32 neurons, ReLU, Adam) &nbsp;|&nbsp;
Trained on 600 student records
</div>
""", unsafe_allow_html=True)
