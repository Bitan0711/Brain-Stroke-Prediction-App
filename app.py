import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load Model
model_data = joblib.load("model.joblib")

st.title("🧠 Brain Stroke Prediction System")

st.write("""
Enter patient details below to estimate stroke risk.
""")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

hypertension = st.selectbox(
    "Hypertension",
    [0,1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0,1]
)

ever_married = st.selectbox(
    "Ever Married",
    ["yes","no"]
)

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self_employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

residence_type = st.selectbox(
    "Residence Type",
    ["Urban","Rural"]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=50.0,
    max_value=300.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    [
        "never smoked",
        "formerly smoked",
        "smokes",
        "unknown"
    ]
)

if st.button("Predict Stroke Risk"):

    sample = {
        "gender": gender.lower(),
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married.lower(),
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status.lower()
    }

    try:

        df = pd.DataFrame([sample])

        encoded_cols = model_data["encoded_cols"]
        numeric_cols = model_data["numeric_cols"]
        preprocessor = model_data["preprocessor"]

        df[encoded_cols] = preprocessor.transform(df)

        X = df[numeric_cols + encoded_cols]

        prediction = model_data["model"].predict(X)[0]

        if prediction == 1:
            st.error("⚠️ High Stroke Risk Detected")
        else:
            st.success("✅ Low Stroke Risk")

    except Exception as e:
        st.error(f"Error: {e}")
