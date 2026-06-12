import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)

# Hide Streamlit menu and footer
st.markdown("""
<style>
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Page background ── */
  .stApp {
    background: linear-gradient(135deg, #0f1923 0%, #162032 60%, #1a2a3a 100%);
    min-height: 100vh;
  }

  /* ── Hero header ── */
  .hero-header {
    text-align: center;
    padding: 2.8rem 1rem 1.6rem;
    margin-bottom: 0.5rem;
  }
  .hero-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #e8f4fd;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem;
  }
  .hero-header .subtitle {
    font-size: 0.95rem;
    color: #7ba7c4;
    font-weight: 400;
    letter-spacing: 0.3px;
  }
  .hero-divider {
    width: 52px;
    height: 3px;
    background: linear-gradient(90deg, #2e9cdb, #56d0c8);
    border-radius: 2px;
    margin: 0.9rem auto 0;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #4a9cc7;
    margin: 1.6rem 0 0.6rem;
    padding-left: 2px;
  }

  /* ── Card wrapper ── */
  .card {
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(6px);
  }

  /* ── Streamlit widget overrides ── */
  div[data-testid="stSelectbox"] > label,
  div[data-testid="stNumberInput"] > label {
    color: #a8c8e0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px;
    margin-bottom: 4px !important;
  }

  div[data-testid="stSelectbox"] > div > div,
  div[data-testid="stNumberInput"] input {
    background: rgba(10, 22, 38, 0.7) !important;
    border: 1px solid rgba(46, 156, 219, 0.25) !important;
    border-radius: 10px !important;
    color: #ddeef8 !important;
    font-size: 0.9rem !important;
  }
  div[data-testid="stSelectbox"] > div > div:focus-within,
  div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(46, 156, 219, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(46, 156, 219, 0.12) !important;
  }

  /* ── Predict button ── */
  div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #1e7fc1, #19a69e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    margin-top: 0.6rem;
  }
  div[data-testid="stButton"] > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
  }
  div[data-testid="stButton"] > button:active {
    transform: translateY(0);
  }

  /* ── Result card ── */
  .result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.4rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-card.high {
    background: linear-gradient(135deg, rgba(210, 50, 50, 0.18), rgba(200, 30, 30, 0.08));
    border: 1px solid rgba(220, 80, 80, 0.35);
  }
  .result-card.low {
    background: linear-gradient(135deg, rgba(30, 170, 110, 0.18), rgba(20, 150, 90, 0.08));
    border: 1px solid rgba(50, 190, 120, 0.35);
  }
  .result-icon { font-size: 2.4rem; margin-bottom: 0.5rem; }
  .result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    font-weight: 400;
    margin-bottom: 0.3rem;
  }
  .result-card.high .result-title { color: #f08080; }
  .result-card.low  .result-title { color: #6cdba8; }
  .result-sub {
    font-size: 0.85rem;
    color: #7ba7c4;
    margin-bottom: 1.2rem;
  }

  /* ── Risk meter ── */
  .meter-wrap { margin: 0.4rem 0 0.2rem; }
  .meter-track {
    height: 12px;
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
  }
  .meter-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
  }
  .meter-fill.high { background: linear-gradient(90deg, #e05555, #f08040); }
  .meter-fill.low  { background: linear-gradient(90deg, #1eaa72, #3fd89a); }
  .meter-pct {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 0.6rem;
    letter-spacing: -1px;
  }
  .meter-pct.high { color: #f08080; }
  .meter-pct.low  { color: #6cdba8; }
  .meter-label {
    font-size: 0.72rem;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #5a8ca8;
    margin-top: 0.1rem;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    font-size: 0.72rem;
    color: #3a6080;
    text-align: center;
    margin-top: 2rem;
    padding: 0 1rem;
    line-height: 1.6;
  }

  /* ── Column gap ── */
  [data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🧠 Brain Stroke Prediction</h1>
  <p class="subtitle">Clinical risk assessment powered by machine learning</p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
model_data = joblib.load("model.joblib")

# ── Form layout ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="section-label">Patient Demographics</p>', unsafe_allow_html=True)
    with st.container():
        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.number_input("Age", min_value=1, max_value=120, value=30)
        ever_married    = st.selectbox("Marital Status", ["yes", "no"],
                                       format_func=lambda x: "Married" if x == "yes" else "Never Married")
        residence_type  = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type       = st.selectbox("Work Type",
                                       ["Private", "Self_employed", "Govt_job", "children", "Never_worked"],
                                       format_func=lambda x: x.replace("_", " ").title())

with col_right:
    st.markdown('<p class="section-label">Clinical Indicators</p>', unsafe_allow_html=True)
    with st.container():
        hypertension      = st.selectbox("Hypertension", [0, 1],
                                          format_func=lambda x: "Yes" if x else "No")
        heart_disease     = st.selectbox("Heart Disease", [0, 1],
                                          format_func=lambda x: "Yes" if x else "No")
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",
                                            min_value=50.0, max_value=300.0, value=100.0)
        bmi               = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        smoking_status    = st.selectbox("Smoking Status",
                                          ["never smoked", "formerly smoked", "smokes", "unknown"],
                                          format_func=lambda x: x.title())

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Predict Stroke Risk", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_clicked:
    sample = {
        "gender":            gender.lower(),
        "age":               age,
        "hypertension":      hypertension,
        "heart_disease":     heart_disease,
        "ever_married":      ever_married.lower(),
        "work_type":         work_type,
        "Residence_type":    residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi":               bmi,
        "smoking_status":    smoking_status.lower()
    }

    try:
        df             = pd.DataFrame([sample])
        encoded_cols   = model_data["encoded_cols"]
        numeric_cols   = model_data["numeric_cols"]
        preprocessor   = model_data["preprocessor"]
        df[encoded_cols] = preprocessor.transform(df)
        X              = df[numeric_cols + encoded_cols]
        prediction     = model_data["model"].predict(X)[0]

        # ── Risk percentage ──
        if hasattr(model_data["model"], "predict_proba"):
            risk_pct = round(model_data["model"].predict_proba(X)[0][1] * 100, 1)
        else:
            risk_pct = 82.0 if prediction == 1 else 18.0

        cls    = "high" if prediction == 1 else "low"
        icon   = "⚠️"   if prediction == 1 else "✅"
        title  = "High Stroke Risk Detected" if prediction == 1 else "Low Stroke Risk"
        subMsg = ("This patient profile shows elevated risk markers. "
                  "Please consult a neurologist promptly."
                  if prediction == 1 else
                  "Current indicators suggest a lower likelihood of stroke. "
                  "Maintain healthy lifestyle habits.")

        _, res_col, _ = st.columns([0.5, 3, 0.5])
        with res_col:
            st.markdown(f"""
            <div class="result-card {cls}">
              <div class="result-icon">{icon}</div>
              <div class="result-title">{title}</div>
              <div class="result-sub">{subMsg}</div>
              <div class="meter-wrap">
                <div class="meter-track">
                  <div class="meter-fill {cls}" style="width:{risk_pct}%"></div>
                </div>
                <div class="meter-pct {cls}">{risk_pct}%</div>
                <div class="meter-label">Estimated Stroke Probability</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<p class="disclaimer">
  This tool is intended for informational purposes only and does not constitute medical advice.<br>
  Always consult a qualified healthcare professional for clinical decisions.
</p>
""", unsafe_allow_html=True)
