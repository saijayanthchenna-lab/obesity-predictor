
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Obesity Category Predictor",
    page_icon="⚖️",
    layout="centered"
)

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model (4).pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        st.error(
            f"Model file not found: {MODEL_PATH.name}. "
            "Check that it exists in your GitHub repository."
        )
        st.stop()

    return joblib.load(MODEL_PATH)

model = load_model()

# Website heading
st.title("⚖️ Obesity Category Predictor")
st.write(
    "Predict an obesity category using your trained "
    "Random Forest machine learning model."
)

st.info(
    "Educational project only — this prediction is "
    "not a medical diagnosis."
)

st.divider()
st.subheader("Enter Personal Details")

# User inputs
with st.form("prediction_form"):
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        max_value=300.0,
        value=65.0
    )

    submitted = st.form_submit_button(
        "Predict Category",
        use_container_width=True
    )

# Make prediction
if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Height": [height],
        "Weight": [weight]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability (%)": probabilities * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Prediction Probabilities")
            st.bar_chart(
                probability_df.set_index("Category")
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
